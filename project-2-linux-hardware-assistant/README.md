# Local Linux Hardware Assistant

A local CLI that answers common CPU, RAM, storage, and GPU questions. A local
Ollama model chooses from a fixed set of read-only inspection tools and explains
their structured results. Hardware data is sent only to the loopback Ollama
service; it is not sent to a cloud model. The assistant does not execute
user-supplied commands.

## Run

Requires Python 3.10 or later, Linux, and a local Ollama installation with a
tool-calling-capable model.

```bash
ollama pull qwen3
python3 assistant.py
```

Run these commands from the repository root.

Ask questions such as “How much RAM do I have?”, “Can I add more RAM?”, “Is my
SSD NVMe?”, “What CPU do I have?”, or “What GPU do I have?”. Type `quit` to exit.
The assistant keeps recent conversation turns so follow-ups such as “and what
about the GPU?” can use the previous context. If Ollama is not already running,
start it in another terminal with `ollama serve`.

The defaults are `OLLAMA_URL=http://127.0.0.1:11434/api/chat` and
`OLLAMA_MODEL=qwen3`. Override them with environment variables if needed.
`OLLAMA_URL` must remain a loopback HTTP URL; the client disables proxies and
redirects.

## Inspection and safety

The assistant has a fixed inspection-tool registry:

| Purpose | Allow-listed tool |
| --- | --- |
| CPU | `lscpu` |
| RAM fallback | `free -b`; prefers read-only `/proc/meminfo` |
| Disk devices | `lsblk --json` with fixed output columns |
| Graphics devices | `lspci` |
| Firmware memory slots/capacity | `dmidecode --type memory` |

The model can select only the named tools below. The application validates every
tool name and rejects arguments; each accepted name maps directly to an existing
`HardwareInspector` method. The model cannot provide commands, paths, or shell
arguments. System commands run as fixed argument arrays with `shell=False`, no
stdin, a timeout, an output-size check, and a fixed locale. The local Ollama
client also limits response size and the number of tool calls.

The optional `dmidecode` inspection is run as the current user only. The
assistant never requests a password, calls `sudo`, or retries with elevated
privileges. Firmware slot and maximum-capacity data may therefore be
unavailable; the assistant is instructed to say so instead of guessing. For
NVMe identification it uses the reported transport or device name; it does not
equate every SSD with NVMe.

## Limitations

* The local model selects tools and phrases answers, but generated text can
  still be mistaken. Tool output is the source of hardware facts; verify
  important upgrade decisions against the laptop's service manual.
* Hardware and firmware reporting varies by kernel, distribution, permissions,
  and installed utilities. Missing tools or fields produce an explicit
  unknown/unavailable response.
* In WSL or a virtual machine, reported values describe the environment visible
  to Linux and may not match the physical host's hardware.
* A reported empty slot and board maximum are firmware data, not a guarantee of
  physical compatibility. Check the laptop model's service manual before
  purchasing memory.

## Tests

The test suite can run on non-Linux development machines; it uses fixtures and
mocked Ollama responses and does not need Ollama or inspect the developer's
hardware:

```bash
python -m unittest -v
```
