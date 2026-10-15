import hashlib
import logging
import math
from base64 import b64encode
from typing import Final

import jwt
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    AssertionIdentity,
    ConsentTeam,
    SubjectIdentity,
    SubjectTokenRefusal,
)
from litellm.proxy._experimental.mcp_server.id_jag_grant import (
    ID_JAG_REQUIRED_CLAIMS,
    REJECTED_ASSERTION,
    TEAM_HEADER,
    AssertionRedemption,
    IdJagPrerequisites,
    basic_client_id,
    identity_from_id_jag,
    jwt_bearer_available,
    select_team,
)
from litellm.proxy._types import IdJagRequestingApp, LiteLLM_JWTAuth
from litellm.proxy.auth.handle_jwt import (
    ID_JAG_TYP,
    AssertionRejection,
    JWKSUnreachableError,
    JWTHandler,
    jwks_unavailable_exception,
)

ISSUER = "https://idp.example.com"
GATEWAY = "https://gateway.example.com"
AUDIENCES = (f"{GATEWAY}/mcp", GATEWAY)
NOW = 1_800_000_000.0
PUBLIC_APP = IdJagRequestingApp(client_id="0oa-public")
CONFIDENTIAL_APP = IdJagRequestingApp(client_id="0oa-confidential", client_secret="s3cret")
EVERY_GATE_HOLDS = {
    "jwt_auth_enabled": True,
    "has_database": True,
    "licensed": True,
    "requesting_apps": (PUBLIC_APP, CONFIDENTIAL_APP),
}
JWKS_DOWN = jwks_unavailable_exception(JWKSUnreachableError("ConnectError fetching the IdP's JWKS after 3 attempts"))


def _claims(**overrides):
    return {
        "iss": ISSUER,
        "sub": "okta-user-1",
        "aud": AUDIENCES[0],
        "exp": NOW + 300,
        "iat": NOW - 5,
        "jti": "jti-1",
        "client_id": PUBLIC_APP.client_id,
        "scope": "mcp",
        **overrides,
    }


def _assertion(typ=ID_JAG_TYP, **claims):
    return jwt.encode(
        _claims(**claims),
        "unused-by-the-fake-verifier-but-long-enough-for-hs256",
        algorithm="HS256",
        headers={"typ": typ},
    )


class _Verifier:
    def __init__(self, result=None, raises=None):
        self.calls = []
        self.result = result
        self.raises = raises

    async def __call__(self, assertion, audience, required_claims):
        self.calls.append((assertion, tuple(audience), tuple(required_claims)))
        if self.raises is not None:
            raise self.raises
        return self.result if self.result is not None else jwt.decode(assertion, options={"verify_signature": False})


class _Resolver:
    def __init__(self, user_id="u1", raises=None, result=None):
        self.calls = []
        self.user_id = user_id
        self.raises = raises
        self.result = result

    async def __call__(self, claims, team_header):
        self.calls.append((dict(claims), team_header))
        if self.raises is not None:
            raise self.raises
        if self.result is not None:
            return self.result
        return None if self.user_id is None else SubjectIdentity(user_id=self.user_id, team_id=team_header)


def _redemption(assertion, client_id=PUBLIC_APP.client_id, client_secret=None, authorization=None, team_id="team-b"):
    return AssertionRedemption(
        assertion=assertion,
        client_id=client_id,
        client_secret=client_secret,
        authorization_header=authorization,
        team_id=team_id,
        audiences=AUDIENCES,
        now=NOW,
    )


async def _identity(redemption, verifier=None, resolver=None, **unmet):
    return await identity_from_id_jag(
        redemption,
        prerequisites=IdJagPrerequisites(**{**EVERY_GATE_HOLDS, **unmet}),
        decode=verifier if verifier is not None else _Verifier(),
        resolve_subject=resolver if resolver is not None else _Resolver(),
    )


@pytest.mark.asyncio
async def test_a_valid_id_jag_names_the_user_its_single_use_key_and_its_scope():
    """The verifier is asked for the gateway's own issuer values as audience and for every claim
    the ID-JAG profile requires, the subject lookup sees the verified claims, and the identity
    carries the team asked for, a replay key derived from iss and jti, a marker TTL that outlives
    the assertion, and the assertion's scope."""
    assertion, verifier, resolver = _assertion(), _Verifier(), _Resolver()
    identity = await _identity(_redemption(assertion), verifier, resolver)
    assert isinstance(identity, AssertionIdentity)
    assert verifier.calls == [(assertion, AUDIENCES, ID_JAG_REQUIRED_CLAIMS)]
    assert {"iss", "sub", "aud", "exp", "iat", "jti", "client_id"} <= set(ID_JAG_REQUIRED_CLAIMS)
    assert resolver.calls == [(_claims(), "team-b")]
    assert (identity.user_id, identity.team_id, identity.scope) == ("u1", "team-b", "mcp")
    assert identity.replay_key == "mcp_gateway_dcr:id_jag:" + hashlib.sha256(f"{ISSUER}\njti-1".encode()).hexdigest()
    assert identity.replay_ttl_seconds == math.ceil(300)
    other_jti = await _identity(_redemption(_assertion(jti="jti-2")), _Verifier(), _Resolver())
    assert isinstance(other_jti, AssertionIdentity)
    assert other_jti.replay_key != identity.replay_key
    teamless = await _identity(_redemption(_assertion(scope=""), team_id=None))
    assert isinstance(teamless, AssertionIdentity)
    assert (teamless.team_id, teamless.scope) == (None, None)


@pytest.mark.asyncio
async def test_a_requesting_app_the_operator_did_not_allowlist_learns_nothing_about_the_assertion():
    verifier, resolver = _Verifier(), _Resolver()
    refused = await _identity(_redemption(_assertion(), client_id="0oa-stranger"), verifier, resolver)
    assert refused == SubjectTokenRefusal(
        error="invalid_client", description="client_id is not an allowlisted requesting app"
    )
    assert verifier.calls == []
    assert resolver.calls == []


def _basic(client_id, secret):
    return "Basic " + b64encode(f"{client_id}:{secret}".encode()).decode()


@pytest.mark.parametrize(
    "client_secret, authorization, accepted",
    [
        (None, None, False),
        ("wrong", None, False),
        ("s3cret", None, True),
        (None, _basic(CONFIDENTIAL_APP.client_id, "s3cret"), True),
        (None, _basic(CONFIDENTIAL_APP.client_id, "wrong"), False),
        (None, _basic(PUBLIC_APP.client_id, "s3cret"), False),
        ("s3cret", _basic(CONFIDENTIAL_APP.client_id, "wrong"), False),
        (None, "Basic not-base64!", False),
        (None, "Bearer s3cret", False),
    ],
)
@pytest.mark.asyncio
async def test_a_confidential_app_must_authenticate_with_its_secret(client_secret, authorization, accepted):
    verifier = _Verifier()
    assertion = _assertion(client_id=CONFIDENTIAL_APP.client_id)
    redemption = _redemption(assertion, CONFIDENTIAL_APP.client_id, client_secret, authorization)
    identity = await _identity(redemption, verifier)
    if accepted:
        assert isinstance(identity, AssertionIdentity)
        return
    assert identity == SubjectTokenRefusal(error="invalid_client", description="client authentication failed")
    assert verifier.calls == []


@pytest.mark.asyncio
async def test_a_public_app_is_identified_by_client_id_alone():
    identity = await _identity(_redemption(_assertion(), client_secret="anything", authorization="Bearer x"))
    assert isinstance(identity, AssertionIdentity)


@pytest.mark.parametrize(
    "assertion, description",
    [
        ("not-a-jwt", "assertion is not a JWT"),
        (_assertion(typ="JWT"), f"assertion typ must be {ID_JAG_TYP}"),
        (_assertion(typ="at+jwt"), f"assertion typ must be {ID_JAG_TYP}"),
    ],
)
@pytest.mark.asyncio
async def test_a_bearer_or_id_token_is_not_an_id_jag(assertion, description):
    verifier = _Verifier()
    refused = await _identity(_redemption(assertion), verifier)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=description)
    assert verifier.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verifier, reason",
    [
        (_Verifier(result=AssertionRejection(reason="Signature verification failed")), "Signature verification"),
        (_Verifier(result=AssertionRejection(reason="Audience doesn't match")), "Audience"),
        (_Verifier(result=AssertionRejection(reason='Token is missing the "jti" claim')), "jti"),
        (_Verifier(result=AssertionRejection(reason="Signature has expired")), "expired"),
        (_Verifier(result=AssertionRejection(reason="iss is not an issuer this gateway trusts")), "iss"),
        (_Verifier(raises=Exception("Validation fails: Invalid audience")), "Invalid audience"),
    ],
)
async def test_an_assertion_the_verifier_rejects_is_refused_with_the_reason_kept_in_the_log(verifier, reason, caplog):
    resolver = _Resolver()
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), verifier, resolver)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert reason not in refused.description
    assert reason in caplog.text
    assert resolver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims, reason",
    [
        ({"client_id": "0oa-confidential"}, "client_id"),
        ({"exp": NOW}, "expired"),
        ({"exp": NOW - 1}, "expired"),
        ({"jti": ""}, "jti"),
        ({"jti": 7}, "jti"),
        ({"iss": None}, "iss"),
    ],
)
async def test_verified_claims_that_do_not_fit_the_profile_are_refused(claims, reason, caplog):
    resolver = _Resolver()
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), _Verifier(result=_claims(**claims)), resolver)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert reason in caplog.text
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_an_assertion_that_names_no_user_the_gateway_knows_cannot_be_redeemed(caplog):
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), resolver=_Resolver(user_id=None))
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert "names no user" in caplog.text
    missing = await _identity(
        _redemption(_assertion()), resolver=_Resolver(raises=ValueError("User doesn't exist in db. 'user_id'=u1"))
    )
    assert missing == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verifier, resolver",
    [
        (_Verifier(raises=JWKS_DOWN), _Resolver()),
        (_Verifier(), _Resolver(raises=HTTPException(status_code=503, detail="database unavailable"))),
    ],
)
async def test_a_gateway_outage_is_reported_as_retryable_not_as_a_bad_assertion(verifier, resolver):
    refused = await _identity(_redemption(_assertion()), verifier, resolver)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "temporarily_unavailable"
    assert "retry" in refused.description


@pytest.mark.parametrize(
    "unmet, description",
    [
        ({"jwt_auth_enabled": False}, "JWT auth is not enabled"),
        ({"has_database": False}, "no database"),
        ({"licensed": False}, "enterprise only"),
        ({"requesting_apps": ()}, "no requesting apps allowlisted"),
    ],
)
@pytest.mark.asyncio
async def test_the_deployment_gates_refuse_before_any_verification(unmet, description):
    verifier = _Verifier()
    refused = await _identity(_redemption(_assertion()), verifier, **unmet)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "unsupported_grant_type"
    assert description in refused.description
    assert verifier.calls == []
    assert IdJagPrerequisites(**{**EVERY_GATE_HOLDS, **unmet}).available is False
    assert IdJagPrerequisites(**EVERY_GATE_HOLDS).available is True


def _running_jwt_handler(litellm_jwtauth):
    handler: Final = JWTHandler()
    if litellm_jwtauth is not None:
        handler.update_environment(prisma_client=None, user_api_key_cache=DualCache(), litellm_jwtauth=litellm_jwtauth)
    return handler


@pytest.mark.parametrize(
    "general_settings, prisma_client, premium_user, litellm_jwtauth, expected",
    [
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), True),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[]), False),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(), False),
        ({"enable_jwt_auth": True}, object(), True, None, False),
        ({}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
        ({"enable_jwt_auth": True}, None, True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
        ({"enable_jwt_auth": True}, object(), False, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
    ],
)
def test_availability_is_read_from_the_running_proxy(
    monkeypatch, general_settings, prisma_client, premium_user, litellm_jwtauth, expected
):
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", general_settings)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", premium_user)
    monkeypatch.setattr("litellm.proxy.proxy_server.jwt_handler", _running_jwt_handler(litellm_jwtauth))
    assert jwt_bearer_available() is expected


@pytest.mark.asyncio
async def test_a_team_the_subject_lookup_cannot_settle_is_refused_before_the_assertion_is_consumed():
    """The resolver's refusal (a user on several teams who named none, or a header naming a team
    they are not on) comes back as is, so the client fixes its request and presents the same
    assertion again instead of finding its jti burned."""
    refusal = SubjectTokenRefusal(error="invalid_request", description=f"send the {TEAM_HEADER} header")
    refused = await _identity(_redemption(_assertion()), resolver=_Resolver(result=refusal))
    assert refused is refusal
    teamed = await _identity(
        _redemption(_assertion()), resolver=_Resolver(result=SubjectIdentity(user_id="u1", team_id="t9"))
    )
    assert isinstance(teamed, AssertionIdentity)
    assert teamed.team_id == "t9"


ENG = ConsentTeam(team_id="team-eng", team_alias="engineering")
OPS = ConsentTeam(team_id="team-ops", team_alias=None)


@pytest.mark.parametrize(
    "team_header, claim_team, teams, expected",
    [
        (None, None, (), None),
        (None, None, (ENG,), "team-eng"),
        ("team-eng", None, (ENG, OPS), "team-eng"),
        ("engineering", None, (ENG, OPS), "team-eng"),
        ("team-ops", "team-eng", (ENG, OPS), "team-ops"),
        (None, "team-ops", (ENG, OPS), "team-ops"),
    ],
)
def test_select_team_prefers_the_header_then_the_assertion_then_the_only_team(team_header, claim_team, teams, expected):
    assert select_team(team_header, claim_team, teams) == expected


@pytest.mark.parametrize(
    "team_header, claim_team, teams, error, mention",
    [
        ("team-sales", None, (ENG, OPS), "invalid_request", TEAM_HEADER),
        ("engineering", None, (), "invalid_request", TEAM_HEADER),
        (None, None, (ENG, OPS), "invalid_request", TEAM_HEADER),
        (None, "team-sales", (ENG, OPS), "invalid_grant", REJECTED_ASSERTION),
        (None, "team-sales", (), "invalid_grant", REJECTED_ASSERTION),
    ],
)
def test_select_team_refuses_a_team_the_user_is_not_on_or_an_unnamed_choice(
    team_header, claim_team, teams, error, mention
):
    refused = select_team(team_header, claim_team, teams)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == error
    assert mention in refused.description


@pytest.mark.parametrize(
    "authorization, expected",
    [
        (None, None),
        ("Bearer abc", None),
        ("Basic not-base64!", None),
        (_basic("0oa-confidential", "s3cret"), "0oa-confidential"),
        ("basic " + b64encode(b"0oa%20x:s3cret").decode(), "0oa x"),
        ("Basic " + b64encode(b"no-separator").decode(), None),
    ],
)
def test_basic_client_id_reads_the_app_from_the_authorization_header(authorization, expected):
    assert basic_client_id(authorization) == expected
