import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from assistant import (
    COMMANDS,
    CommandResult,
    HardwareInspector,
    OllamaError,
    OllamaToolAssistant,
    answer_question,
    parse_dmi_memory,
    parse_free,
    parse_lsblk,
    parse_lscpu,
    parse_lspci_gpus,
    parse_meminfo,
    run_allowlisted_command,
)


class FakeOllamaResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _ = exc_type, exc_value, traceback
        return False

    def read(self, limit):
        return self.body[:limit]


class ParserTests(unittest.TestCase):
    def test_parse_lscpu_fields(self):
        result = parse_lscpu(
            "Architecture:                    x86_64\n"
            "CPU(s):                          8\n"
            "Model name:                      Example CPU\n"
            "Thread(s) per core:              2\n"
        )
        self.assertEqual(result["architecture"], "x86_64")
        self.assertEqual(result["logical_cpus"], "8")
        self.assertEqual(result["model"], "Example CPU")

    def test_parse_meminfo_kilobytes(self):
        result = parse_meminfo("MemTotal:       16384000 kB\nMemAvailable:    8000000 kB\n")
        self.assertEqual(result["MemTotal"], 16384000 * 1024)
        self.assertEqual(result["MemAvailable"], 8000000 * 1024)

    def test_parse_free_fallback(self):
        result = parse_free(
            "               total        used        free      shared  buff/cache   available\n"
            "Mem:     16000000000 4000000000 6000000000 100000000 6000000000 11000000000\n"
        )
        self.assertEqual(result["total"], 16000000000)
        self.assertEqual(result["available"], 11000000000)

    def test_parse_lsblk_and_detect_nvme_without_guessing_ssd_bus(self):
        output = json.dumps(
            {
                "blockdevices": [
                    {"name": "nvme0n1", "type": "disk", "size": 512000000000, "tran": "nvme"},
                    {"name": "sda", "type": "disk", "size": 1000000000000, "rota": False},
                    {"name": "sda1", "type": "part", "size": 1000000000000},
                ]
            }
        )
        result = parse_lsblk(output)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["media"], "NVMe")
        self.assertEqual(result[1]["media"], "SSD (bus type unknown)")

    def test_parse_gpus_only(self):
        output = (
            "00:02.0 VGA compatible controller: Example Integrated Graphics\n"
            "01:00.0 3D controller: Example Discrete GPU\n"
            "00:14.0 USB controller: Example USB\n"
        )
        self.assertEqual(len(parse_lspci_gpus(output)), 2)

    def test_parse_dmi_reports_empty_slots_and_capacity(self):
        output = (
            "Physical Memory Array\n\tMaximum Capacity: 64 GB\n"
            "Memory Device\n\tSize: 16 GB\n\tLocator: DIMM_A1\n"
            "Memory Device\n\tSize: No Module Installed\n\tLocator: DIMM_B1\n"
        )
        result = parse_dmi_memory(output)
        self.assertEqual(result["installed_count"], 1)
        self.assertEqual(result["empty_slots"], 1)
        self.assertEqual(result["maximum_capacity"], "64 GB")


class SafetyTests(unittest.TestCase):
    def test_only_fixed_commands_and_arguments_are_run_without_shell(self):
        with patch("assistant.shutil.which", return_value="C:\\tools\\lscpu.exe"), patch(
            "assistant.subprocess.run",
            return_value=type(
                "ProcessResult", (), {"stdout": "CPU(s): 4", "stderr": "", "returncode": 0}
            )(),
        ) as run:
            self.assertEqual(run_allowlisted_command("cpu").stdout, "CPU(s): 4")
        arguments = run.call_args.args[0]
        options = run.call_args.kwargs
        self.assertEqual(arguments, ["C:\\tools\\lscpu.exe"])
        self.assertFalse(options["shell"])
        self.assertEqual(options["stdin"], -3)
        with self.assertRaises(ValueError):
            run_allowlisted_command("arbitrary")

    def test_all_allowlisted_commands_are_read_only_inspection_tools(self):
        self.assertEqual(
            set(COMMANDS),
            {"cpu", "memory", "storage", "pci", "memory_modules"},
        )
        self.assertEqual(COMMANDS["memory_modules"], ("dmidecode", ("--type", "memory")))

    def test_upgrade_answer_does_not_escalate_privileges(self):
        def fake_runner(name):
            if name == "memory_modules":
                return CommandResult(error="dmidecode: permission denied")
            return CommandResult()

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "meminfo").write_text("MemTotal: 16777216 kB\n", encoding="ascii")
            inspector = HardwareInspector(fake_runner, Path(directory))
            response, topic = answer_question("Can I add more RAM?", inspector)
        self.assertEqual(topic, "memory")
        self.assertIn("available slots or maximum supported capacity", response)
        self.assertIn("did not attempt to run with elevated privileges", response)


class AssistantTests(unittest.TestCase):
    def test_answers_ram_and_follows_up_about_upgrade(self):
        def fake_runner(name):
            if name == "memory_modules":
                return CommandResult(
                    stdout=(
                        "Physical Memory Array\n\tMaximum Capacity: 64 GB\n"
                        "Memory Device\n\tSize: 16 GB\n"
                        "Memory Device\n\tSize: No Module Installed\n"
                    )
                )
            return CommandResult()

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "meminfo").write_text("MemTotal: 16777216 kB\n", encoding="ascii")
            inspector = HardwareInspector(fake_runner, Path(directory))
            first, topic = answer_question("How much RAM do I have?", inspector)
            follow_up, next_topic = answer_question("And can I add more?", inspector, topic)
        self.assertIn("16.0 GiB", first)
        self.assertIn("1 populated module(s) and 1 empty slot(s)", follow_up)
        self.assertIn("64 GB", follow_up)
        self.assertEqual(topic, next_topic)

    def test_unknown_question_does_not_run_tools(self):
        def forbidden_runner(_):
            self.fail("An unrelated question must not inspect hardware")

        response, topic = answer_question("What is the weather?", HardwareInspector(forbidden_runner))
        self.assertIn("I can check CPU, RAM", response)
        self.assertIsNone(topic)

    def test_gpu_and_storage_questions_use_the_matching_tool(self):
        calls = []

        def fake_runner(name):
            calls.append(name)
            if name == "pci":
                return CommandResult(stdout="00:02.0 VGA compatible controller: Example GPU\n")
            if name == "storage":
                return CommandResult(
                    stdout=json.dumps(
                        {
                            "blockdevices": [
                                {"name": "nvme0n1", "type": "disk", "size": 1000000000, "tran": "nvme"}
                            ]
                        }
                    )
                )
            return CommandResult()

        inspector = HardwareInspector(fake_runner)
        gpu, _ = answer_question("What GPU do I have?", inspector)
        storage, _ = answer_question("Is my SSD NVMe?", inspector)
        self.assertIn("Example GPU", gpu)
        self.assertIn("NVMe", storage)
        self.assertEqual(calls, ["pci", "storage"])


class OllamaToolTests(unittest.TestCase):
    def test_valid_tool_call_runs_only_the_selected_inspector_and_uses_its_result(self):
        inspector = Mock(spec=HardwareInspector)
        inspector.memory.return_value = {"total": 16 * 1024**3, "available": 8 * 1024**3}
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_memory_info",
                                "arguments": {},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "This system has 16 GiB of RAM.",
                }
            },
        ]
        requests = []
        timeouts = []

        def fake_open(request, timeout):
            requests.append(json.loads(request.data))
            timeouts.append(timeout)
            return FakeOllamaResponse(responses.pop(0))

        with patch("assistant.LOCAL_OLLAMA_OPENER.open", side_effect=fake_open):
            answer = OllamaToolAssistant().ask("How much RAM do I have?", inspector)

        self.assertEqual(answer, "This system has 16 GiB of RAM.")
        self.assertEqual(timeouts, [90, 90])
        inspector.memory.assert_called_once_with()
        inspector.cpu.assert_not_called()
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[0]["tools"]],
            [
                "get_cpu_info",
                "get_memory_info",
                "get_memory_topology",
                "get_storage_devices",
                "get_graphics_devices",
            ],
        )
        tool_messages = [
            message
            for message in requests[1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(tool_messages[0]["tool_name"], "get_memory_info")
        self.assertEqual(
            json.loads(tool_messages[0]["content"]),
            {"total": 16 * 1024**3, "available": 8 * 1024**3},
        )

    def test_unknown_tool_is_rejected_without_inspecting_hardware(self):
        inspector = Mock(spec=HardwareInspector)
        response = FakeOllamaResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_shell_command",
                                "arguments": {"command": "id"},
                            }
                        }
                    ],
                }
            }
        )

        with patch("assistant.LOCAL_OLLAMA_OPENER.open", return_value=response):
            with self.assertRaisesRegex(OllamaError, "unapproved inspection tool"):
                OllamaToolAssistant().ask("Check my system", inspector)

        self.assertEqual(inspector.method_calls, [])

    def test_tool_arguments_are_rejected_without_inspecting_hardware(self):
        inspector = Mock(spec=HardwareInspector)
        response = FakeOllamaResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_cpu_info",
                                "arguments": {"command": "id"},
                            }
                        }
                    ],
                }
            }
        )

        with patch("assistant.LOCAL_OLLAMA_OPENER.open", return_value=response):
            with self.assertRaisesRegex(OllamaError, "does not accept arguments"):
                OllamaToolAssistant().ask("Check my CPU", inspector)

        self.assertEqual(inspector.method_calls, [])

    def test_ollama_endpoint_must_be_loopback(self):
        with self.assertRaisesRegex(ValueError, "local Ollama"):
            OllamaToolAssistant(endpoint="http://example.com/api/chat")


if __name__ == "__main__":
    unittest.main()
