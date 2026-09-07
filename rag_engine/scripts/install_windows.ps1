# Installs a Windows Task Scheduler task that runs the local search daemon
# (app/tray.py, PROFILE=local) at every login. OFF by default -- this script does
# nothing until you run it yourself. See the end of its output for how to undo it.
#
# Usage (from an ordinary PowerShell prompt, no admin rights required for a
# per-user logon task):
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$TaskName = "RagSearchLocalDaemon"
$ScriptPath = Join-Path $RepoRoot "scripts\run_local_daemon.py"

# Re-registering should replace, not duplicate, a previous install.
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Task Scheduler actions don't take a per-task environment variable directly, so
# PROFILE=local is set via a cmd.exe wrapper around the actual command.
$innerCommand = "set PROFILE=local && `"$PythonExe`" `"$ScriptPath`""
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $innerCommand" -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Runs the RagSearch local search daemon (PROFILE=local) at login." | Out-Null

Write-Host "Installed scheduled task '$TaskName' to run at every login:"
Write-Host "  PROFILE=local $PythonExe $ScriptPath"
Write-Host "  (working directory: $RepoRoot)"
Write-Host ""
Write-Host "To undo:"
Write-Host "  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
