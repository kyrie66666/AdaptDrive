# Historical Server-10 Incidents

This file is historical context only. It is not a runtime configuration and
must not be copied into a new server's environment.

## Recorded symptoms

- GPU 0 was unreliable on the former server. That restriction is retired; a
  new server must enumerate all visible GPUs and may use GPU 0 when healthy.
- CARLA could start but the client could not establish the expected RPC
  connection. Treat this as a host, port, firewall, or process-lifecycle issue,
  not as evidence for a fixed GPU or display setting.
- Old CARLA notes used `DISPLAY=:99`, a fixed Vulkan adapter, and
  server-specific ports. These values were local workarounds and are not
  AdaptDrive defaults.
- Historical source and asset roots under `/data9_server7`,
  `/data3_server8`, `/home/tmp2`, `/opt/data/private/project`, and
  `/home/deeplearning` are provenance only. They must not be embedded in new
  runtime files.

## Diagnostic use

Read this file only when comparing a new failure with the old incident. First
collect the new server's GPU table, CARLA process command line, RPC/TM ports,
display or EGL state, and the final few lines of the CARLA log. Do not
bulk-read old logs.

## Portability rule

The new server's bootstrap must discover `CARLA_ROOT`, GPU identity, CUDA
visibility, Vulkan/EGL availability, `DISPLAY` or headless mode, and free ports
at runtime. A historical value may be used only after a new probe proves it is
valid on the current host.
