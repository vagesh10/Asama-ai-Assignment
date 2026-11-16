import io
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import patch

from app import (
    APIError,
    LOCAL_OPENER,
    OllamaClient,
    OllamaEmbeddingClient,
    Platform,
    create_server,
)


class FakeDraftClient:
    def __init__(self, draft="A relevant offer; no unsupported terms."):
        self.response = draft
        self.calls = []

    def draft(self, post, agent):
        self.calls.append((post, agent))
        return self.response


class FakeEmbeddingClient:
    model = "test-embedding-model"

    def embed(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "diwali delivery offer" in normalized or "digital payment" in normalized:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakeWebhookTransport:
    def __init__(self, status=204):
        self.status = status
        self.calls = []

    def send(self, url, body, headers):
        self.calls.append((url, body, headers))
        return self.status


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.platform = Platform(os.path.join(self.temp_dir.name, "test.db"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def register_verified(self, name, interests):
        agent = self.platform.register_agent(name, name, interests)
        self.platform.verify_agent(agent["id"])
        return agent

    def test_routes_by_interest_and_excludes_unrelated_agent(self):
        publisher = self.register_verified("Swiggy", ["food"])
        phonepe = self.register_verified("PhonePe", ["payments", "cashback"])
        unrelated = self.register_verified("Bajaj Auto", ["motorcycles"])

        post = self.platform.create_post(
            publisher["id"],
            "Diwali delivery offer",
            "5% off all food deliveries. Pay with UPI for extra savings.",
            ["food", "payments"],
        )

        notifications = self.platform.list_notifications(phonepe["id"])
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["post_id"], post["id"])
        self.assertGreaterEqual(notifications[0]["score"], 0.35)
        self.assertEqual(notifications[0]["reasons"], ["tag match: payments"])
        self.assertEqual(self.platform.list_notifications(unrelated["id"]), [])

    def test_requires_notification_to_comment_and_marks_response(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        stranger = self.register_verified("HDFC", ["banking"])
        post = self.platform.create_post(
            publisher["id"], "UPI offer", "Extra cashback on UPI purchases", ["payments"]
        )

        comment = self.platform.add_comment(recipient["id"], post["id"], "Add a card offer too.")
        self.assertEqual(comment["author"], "PhonePe")
        with self.assertRaisesRegex(Exception, "Only an agent notified"):
            self.platform.add_comment(stranger["id"], post["id"], "Unsolicited comment")
        self.assertEqual(self.platform.list_notifications(recipient["id"])[0]["status"], "responded")

    def test_keyword_match_below_threshold_does_not_notify(self):
        publisher = self.register_verified("Swiggy", ["food"])
        candidate = self.register_verified("Travel", ["food delivery"])
        self.platform.set_subscriptions(candidate["id"], ["food delivery"], 0.6)

        self.platform.create_post(publisher["id"], "Food nearby", "Find food now", [])

        self.assertEqual(self.platform.list_notifications(candidate["id"]), [])

    def test_semantic_similarity_routes_related_terms_without_shared_keywords(self):
        platform = Platform(
            os.path.join(self.temp_dir.name, "semantic.db"),
            FakeEmbeddingClient(),
        )
        publisher = platform.register_agent("Swiggy", "Swiggy", ["food"])
        platform.verify_agent(publisher["id"])
        payments = platform.register_agent("PhonePe", "PhonePe", ["digital payment partnerships"])
        unrelated = platform.register_agent("Bajaj Auto", "Bajaj Auto", ["motorcycles"])
        platform.verify_agent(payments["id"])
        platform.verify_agent(unrelated["id"])

        post = platform.create_post(
            publisher["id"],
            "Diwali delivery offer",
            "Festive meals available this week",
            [],
        )

        notifications = platform.list_notifications(payments["id"])
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["post_id"], post["id"])
        self.assertIn("semantic similarity: 1.000", notifications[0]["reasons"])
        self.assertEqual(platform.list_notifications(unrelated["id"]), [])

    def test_router_caps_delivery_to_five_highest_scoring_agents(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipients = [
            self.register_verified(f"Agent {index}", ["payments"])
            for index in range(6)
        ]

        self.platform.create_post(
            publisher["id"], "Offer", "Delivery discount", ["payments"]
        )

        notifications = [
            self.platform.list_notifications(agent["id"])
            for agent in recipients
        ]
        self.assertEqual(sum(bool(inbox) for inbox in notifications), 5)

    def test_feedback_is_scoped_to_recipient_and_updates_routing_metrics(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        stranger = self.register_verified("HDFC", ["banking"])
        post = self.platform.create_post(
            publisher["id"], "Diwali offer", "Food delivery discount", ["payments"]
        )
        notification = self.platform.list_notifications(recipient["id"])[0]

        with self.assertRaisesRegex(APIError, "Notification not found"):
            self.platform.submit_feedback(stranger["id"], notification["id"], "relevant")
        with self.assertRaisesRegex(APIError, "feedback must be"):
            self.platform.submit_feedback(recipient["id"], notification["id"], {"relevant": True})

        feedback = self.platform.submit_feedback(
            recipient["id"], notification["id"], "irrelevant"
        )
        self.assertEqual(feedback["feedback"], "irrelevant")
        self.platform.submit_feedback(recipient["id"], notification["id"], "relevant")
        self.platform.add_comment(recipient["id"], post["id"], "A relevant comment.")

        metrics = self.platform.routing_metrics(recipient["id"])
        self.assertEqual(metrics["notifications"], 1)
        self.assertEqual(metrics["responded"], 1)
        self.assertEqual(metrics["engagement_rate"], 1.0)
        self.assertEqual(metrics["explicit_feedback"], {"relevant": 1, "irrelevant": 0})
        self.assertEqual(metrics["score_bands"][0]["notifications"], 1)
        self.assertEqual(
            self.platform.list_notifications(recipient["id"])[0]["feedback"],
            "relevant",
        )

    def test_metrics_with_no_notifications_are_well_defined(self):
        agent = self.register_verified("PhonePe", ["payments"])

        metrics = self.platform.routing_metrics(agent["id"])

        self.assertEqual(metrics["notifications"], 0)
        self.assertEqual(metrics["responded"], 0)
        self.assertIsNone(metrics["engagement_rate"])
        self.assertEqual(metrics["score_bands"], [])

    def test_webhook_delivery_is_signed_and_marks_outbox_delivered(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        master_key = "local-test-master-key-with-32-bytes!"
        with patch.dict(os.environ, {"AGENT_NETWORK_WEBHOOK_MASTER_KEY": master_key}), patch.object(
            Platform,
            "_validate_webhook_url",
            return_value=("hooks.example.com", 443, "/events", ["203.0.113.8"]),
        ):
            self.platform.set_webhook(recipient["id"], "https://hooks.example.com/events")
            post = self.platform.create_post(
                publisher["id"], "Diwali offer", "Food delivery discount", ["payments"]
            )
            transport = FakeWebhookTransport()

            self.assertTrue(self.platform.process_webhook_delivery(transport))

            secret = bytes.fromhex(self.platform.webhook_secret(recipient["id"]))

        self.assertEqual(len(transport.calls), 1)
        url, body, headers = transport.calls[0]
        self.assertEqual(url, "https://hooks.example.com/events")
        payload = json.loads(body)
        self.assertEqual(payload["event"], "post.relevant")
        self.assertEqual(payload["post"]["id"], post["id"])
        signed_message = headers["X-Agent-Network-Timestamp"].encode() + b"." + body
        expected = hmac.new(secret, signed_message, hashlib.sha256).hexdigest()
        self.assertEqual(headers["X-Agent-Network-Signature"], f"sha256={expected}")
        with self.platform._connect() as db:
            row = db.execute(
                """
                SELECT d.status, d.attempts, d.last_error
                FROM webhook_deliveries d JOIN notifications n ON n.id = d.notification_id
                WHERE n.agent_id = ?
                """,
                (recipient["id"],),
            ).fetchone()
        self.assertEqual(dict(row), {"status": "delivered", "attempts": 1, "last_error": None})

    def test_webhook_non_success_is_retried_and_url_validation_rejects_private_targets(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        with patch.dict(
            os.environ,
            {"AGENT_NETWORK_WEBHOOK_MASTER_KEY": "local-test-master-key-with-32-bytes!"},
        ), patch.object(
            Platform,
            "_validate_webhook_url",
            return_value=("hooks.example.com", 443, "/events", ["203.0.113.8"]),
        ):
            self.platform.set_webhook(recipient["id"], "https://hooks.example.com/events")
            self.platform.create_post(
                publisher["id"], "Diwali offer", "Food delivery discount", ["payments"]
            )
            self.platform.process_webhook_delivery(FakeWebhookTransport(status=503))

        with self.platform._connect() as db:
            row = db.execute(
                """
                SELECT d.status, d.attempts, d.last_error
                FROM webhook_deliveries d JOIN notifications n ON n.id = d.notification_id
                WHERE n.agent_id = ?
                """,
                (recipient["id"],),
            ).fetchone()
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["last_error"], "Webhook returned HTTP 503")
        with self.assertRaisesRegex(APIError, "public HTTPS URL"):
            Platform._validate_webhook_url("http://hooks.example.com/events")
        with self.assertRaisesRegex(APIError, "public DNS hostname"):
            Platform._validate_webhook_url("https://127.0.0.1/events")

    def test_webhook_configuration_requires_master_key(self):
        agent = self.register_verified("PhonePe", ["payments"])

        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            APIError, "Webhook signing is not configured"
        ):
            self.platform.set_webhook(agent["id"], "https://hooks.example.com/events")

    def test_migrates_existing_subscriptions_table_for_embeddings(self):
        database_path = os.path.join(self.temp_dir.name, "legacy.db")
        db = sqlite3.connect(database_path)
        try:
            db.executescript(
                """
                CREATE TABLE agents (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    company TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    verified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE subscriptions (
                    agent_id INTEGER NOT NULL,
                    interest TEXT NOT NULL,
                    min_score REAL NOT NULL DEFAULT 0.35,
                    PRIMARY KEY (agent_id, interest)
                );
                """
            )
        finally:
            db.close()

        Platform(database_path, FakeEmbeddingClient())

        db = sqlite3.connect(database_path)
        try:
            columns = {row[1] for row in db.execute("PRAGMA table_info(subscriptions)")}
        finally:
            db.close()
        self.assertIn("embedding", columns)
        self.assertIn("embedding_model", columns)

    def test_draft_is_limited_to_notified_verified_agent_and_is_not_published(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        stranger = self.register_verified("Bajaj Auto", ["motorcycles"])
        post = self.platform.create_post(
            publisher["id"], "Diwali offer", "Discount on food deliveries", ["payments"]
        )
        client = FakeDraftClient()

        result = self.platform.draft_comment(recipient["id"], post["id"], client)

        self.assertEqual(result["draft"], client.response)
        self.assertEqual(client.calls[0][0]["company"], "Swiggy")
        self.assertEqual(client.calls[0][1]["company"], "PhonePe")
        self.assertEqual(self.platform.get_post(post["id"])["comments"], [])
        with self.assertRaisesRegex(APIError, "Only an agent notified"):
            self.platform.draft_comment(stranger["id"], post["id"], client)
        self.assertEqual(len(client.calls), 1)

    def test_ollama_url_must_target_local_chat_api(self):
        with self.assertRaisesRegex(ValueError, "local Ollama"):
            OllamaClient(endpoint="http://example.com:11434/api/chat")
        with self.assertRaisesRegex(ValueError, "local Ollama"):
            OllamaClient(endpoint="http://localhost:11434/other")

    def test_embedding_url_must_target_local_ollama(self):
        with self.assertRaisesRegex(ValueError, "local Ollama"):
            OllamaEmbeddingClient(
                endpoint="http://example.com:11434/api/embed",
                model="nomic-embed-text",
            )
        with self.assertRaisesRegex(ValueError, "OLLAMA_EMBEDDING_MODEL"):
            OllamaEmbeddingClient(model="")

    def test_ollama_embedding_request_uses_fixed_model_and_texts(self):
        response = io.BytesIO(b'{"embeddings":[[3.0,4.0]]}')
        client = OllamaEmbeddingClient(
            endpoint="http://127.0.0.1:11434/api/embed",
            model="nomic-embed-text",
        )
        with patch.object(LOCAL_OPENER, "open", return_value=response) as open_request:
            embeddings = client.embed(["Diwali delivery offer"])

        self.assertEqual(embeddings, [[3.0, 4.0]])
        request = open_request.call_args.args[0]
        self.assertEqual(
            json.loads(request.data),
            {"model": "nomic-embed-text", "input": ["Diwali delivery offer"]},
        )

    def test_ollama_request_uses_separated_untrusted_post_data(self):
        response = io.BytesIO(b'{"message":{"content":"A grounded reply."}}')
        client = OllamaClient(endpoint="http://127.0.0.1:11434/api/chat", model="test-model")
        with patch.object(LOCAL_OPENER, "open", return_value=response) as open_request:
            draft = client.draft(
                {"company": "Swiggy", "title": "Offer", "body": "5% off", "tags": ["food"]},
                {"company": "PhonePe", "interests": ["payments"]},
            )

        self.assertEqual(draft, "A grounded reply.")
        request = open_request.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertIn("untrusted data", payload["messages"][0]["content"])
        self.assertEqual(
            json.loads(payload["messages"][1]["content"])["post"]["body"],
            "5% off",
        )

    def test_model_can_decline_to_reply(self):
        publisher = self.register_verified("Swiggy", ["food"])
        recipient = self.register_verified("PhonePe", ["payments"])
        post = self.platform.create_post(
            publisher["id"], "Diwali offer", "Discount on food deliveries", ["payments"]
        )

        result = self.platform.draft_comment(recipient["id"], post["id"], FakeDraftClient("NO_REPLY"))

        self.assertIsNone(result["draft"])
        self.assertEqual(self.platform.get_post(post["id"])["comments"], [])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.draft_client = FakeDraftClient()
        self.server = create_server(
            port=0,
            database_path=os.path.join(self.temp_dir.name, "http.db"),
            draft_client=self.draft_client,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp_dir.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection(self.host, self.port)
        encoded = json.dumps(body).encode() if body is not None else None
        request_headers = dict(headers or {})
        if encoded:
            request_headers["Content-Type"] = "application/json"
        connection.request(method, path, body=encoded, headers=request_headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def test_registration_verification_publish_and_poll_flow(self):
        status, publisher = self.request(
            "POST",
            "/agents",
            {"name": "Swiggy", "company": "Swiggy", "interests": ["food"]},
        )
        self.assertEqual(status, 201)
        self.assertFalse(publisher["verified"])
        recipient_status, recipient = self.request(
            "POST",
            "/agents",
            {"name": "PhonePe", "company": "PhonePe", "interests": ["payments"]},
        )
        self.assertEqual(recipient_status, 201)
        auth = {"Authorization": f"Bearer {publisher['token']}"}
        recipient_auth = {"Authorization": f"Bearer {recipient['token']}"}

        status, _ = self.request(
            "POST",
            "/posts",
            {"title": "Offer", "body": "5% off food", "tags": ["food"]},
            auth,
        )
        self.assertEqual(status, 403)

        os.environ["AGENT_NETWORK_ADMIN_KEY"] = "test-admin"
        try:
            for agent in (publisher, recipient):
                status, _ = self.request(
                    "POST",
                    f"/admin/agents/{agent['id']}/verify",
                    headers={"X-Admin-Key": "test-admin"},
                )
                self.assertEqual(status, 200)
        finally:
            os.environ.pop("AGENT_NETWORK_ADMIN_KEY", None)

        status, post = self.request(
            "POST",
            "/posts",
            {"title": "Offer", "body": "5% off deliveries", "tags": ["payments"]},
            auth,
        )
        self.assertEqual(status, 201)
        self.assertEqual(post["title"], "Offer")
        status, inbox = self.request("GET", "/notifications", headers=recipient_auth)
        self.assertEqual(status, 200)
        self.assertEqual(len(inbox["notifications"]), 1)
        status, draft = self.request(
            "POST", f"/posts/{post['id']}/draft-comment", headers=recipient_auth
        )
        self.assertEqual(status, 200)
        self.assertEqual(draft["draft"], self.draft_client.response)
        self.assertEqual(self.draft_client.calls[0][0]["title"], "Offer")
        status, post_after_draft = self.request(
            "POST",
            f"/posts/{post['id']}/comments",
            {"body": draft["draft"]},
            recipient_auth,
        )
        self.assertEqual(status, 201)
        self.assertEqual(post_after_draft["body"], draft["draft"])
        self.assertEqual(post_after_draft["author"], "PhonePe")

    def test_rejects_bad_json_body(self):
        connection = HTTPConnection(self.host, self.port)
        connection.request(
            "POST",
            "/agents",
            body=b"{bad",
            headers={"Content-Type": "application/json", "Content-Length": "4"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.read())["error"], "Request body must be valid JSON")
        connection.close()

    def test_feedback_and_metrics_http_endpoints(self):
        publisher = self.server.RequestHandlerClass.platform.register_agent(
            "Swiggy", "Swiggy", ["food"]
        )
        recipient = self.server.RequestHandlerClass.platform.register_agent(
            "PhonePe", "PhonePe", ["payments"]
        )
        self.server.RequestHandlerClass.platform.verify_agent(publisher["id"])
        self.server.RequestHandlerClass.platform.verify_agent(recipient["id"])
        post = self.server.RequestHandlerClass.platform.create_post(
            publisher["id"], "Diwali offer", "Food delivery discount", ["payments"]
        )
        inbox = self.server.RequestHandlerClass.platform.list_notifications(recipient["id"])
        headers = {"Authorization": f"Bearer {recipient['token']}"}

        status, feedback = self.request(
            "POST",
            f"/notifications/{inbox[0]['id']}/feedback",
            {"feedback": "relevant"},
            headers,
        )
        self.assertEqual(status, 200)
        self.assertEqual(feedback["feedback"], "relevant")
        status, metrics = self.request(
            "GET", "/agents/me/routing-metrics", headers=headers
        )
        self.assertEqual(status, 200)
        self.assertEqual(metrics["explicit_feedback"]["relevant"], 1)
        self.assertEqual(metrics["notifications"], 1)
        self.assertEqual(post["id"], inbox[0]["post_id"])

    def test_webhook_settings_and_secret_http_endpoints(self):
        agent = self.server.platform.register_agent(
            "PhonePe", "PhonePe", ["payments"]
        )
        headers = {"Authorization": f"Bearer {agent['token']}"}
        with patch.dict(
            os.environ,
            {"AGENT_NETWORK_WEBHOOK_MASTER_KEY": "local-test-master-key-with-32-bytes!"},
        ), patch.object(
            Platform,
            "_validate_webhook_url",
            return_value=("hooks.example.com", 443, "/events", ["203.0.113.8"]),
        ):
            status, configured = self.request(
                "PUT",
                "/agents/me/webhook",
                {"webhook_url": "https://hooks.example.com/events"},
                headers,
            )
            self.assertEqual(status, 200)
            self.assertEqual(
                configured["webhook_url"], "https://hooks.example.com/events"
            )
            status, signing_key = self.request(
                "GET", "/agents/me/webhook-secret", headers=headers
            )
        self.assertEqual(status, 200)
        self.assertEqual(signing_key["algorithm"], "HMAC-SHA256")
        self.assertEqual(signing_key["agent_id"], agent["id"])
        self.assertEqual(len(signing_key["secret"]), 64)


if __name__ == "__main__":
    unittest.main()
