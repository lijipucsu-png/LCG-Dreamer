param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,

    [Parameter(Mandatory = $true)]
    [string]$MetricsLog,

    [Parameter(Mandatory = $true)]
    [string]$AlertPath,

    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$criteria = @(
    @{ max_gap = 15.0; shield = 0.08; string = 1.30 },
    @{ max_gap = 18.0; shield = 0.15; string = 1.50 },
    @{ max_gap = 22.0; shield = 0.30; string = 1.80 },
    @{ max_gap = 20.0; shield = 0.25; string = 1.60 },
    @{ max_gap = 24.0; shield = 0.30; string = 1.80 },
    @{ max_gap = 30.0; shield = 0.35; string = 2.00 },
    @{ max_gap = 30.0; shield = 0.35; string = 2.00 },
    @{ max_gap = 32.0; shield = 0.35; string = 2.00 }
)
$initialLogLineCount = if (Test-Path -LiteralPath $MetricsLog) {
    @(Get-Content -LiteralPath $MetricsLog).Count
}
else { 0 }
$watchStartedAt = Get-Date

while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
    if (-not (Test-Path -LiteralPath $MetricsLog)) {
        continue
    }

    $logFile = Get-Item -LiteralPath $MetricsLog
    if (
        ((Get-Date) - $watchStartedAt).TotalMinutes -gt 20.0 -and
        ((Get-Date) - $logFile.LastWriteTime).TotalMinutes -gt 20.0
    ) {
        $reasons = @("metrics log has not advanced for 20 minutes")
        $records = @()
    }
    else {
        $parsedRecords = @(
            Get-Content -LiteralPath $MetricsLog |
                Select-Object -Skip $initialLogLineCount |
                ForEach-Object {
                    try { $_ | ConvertFrom-Json } catch { $null }
                } |
                Where-Object {
                    $null -ne $_ -and
                    $_.PSObject.Properties.Name -contains "training/episode_max_gap"
                }
        )
        # A resumed run can legitimately rewrite the same training step.  Do
        # not interpret duplicate rows from repeated checkpoint recovery as
        # consecutive failures.
        $recordsByStep = @{}
        foreach ($record in $parsedRecords) {
            $recordsByStep[[string][int]$record.step] = $record
        }
        $records = @(
            $recordsByStep.Values |
                Sort-Object { [int]$_.step } |
                Select-Object -Last 3
        )
        if ($records.Count -eq 0) {
            continue
        }
        $latest = $records[-1]
        if ([int]$latest.step -lt 20000) {
            continue
        }
        $actorReplayWarmingUp = (
            $latest.PSObject.Properties.Name -contains
            "training/replay_steps" -and
            [double]$latest.'training/replay_steps' -lt 10000.0
        )

        $stage = [Math]::Min(
            [int]$latest.'training/curriculum_stage',
            $criteria.Count - 1
        )
        $limit = $criteria[$stage]
        $reasons = @()
        $failures = @($records | ForEach-Object { [double]$_.'training/failure' })
        $collisions = @($records | ForEach-Object { [double]$_.'training/collision' })
        $maxGaps = @($records | ForEach-Object { [double]$_.'training/episode_max_gap' })
        $shields = @($records | ForEach-Object { [double]$_.'training/episode_shield' })
        $strings = @($records | ForEach-Object { [double]$_.'training/episode_string' })

        $invalidTerminalSuccess = @(
            $records | Where-Object {
                [double]$_.'training/success' -ge 0.5 -and
                (
                    -not ($_.PSObject.Properties.Name -contains
                        'training/terminal_platoon_active_count') -or
                    -not ($_.PSObject.Properties.Name -contains
                        'training/terminal_connected_pair_count') -or
                    [double]$_.'training/terminal_platoon_active_count' -ne 6.0 -or
                    [double]$_.'training/terminal_connected_pair_count' -ne 5.0
                )
            }
        )
        if ($invalidTerminalSuccess.Count -gt 0) {
            $reasons += "success reported without the complete six-truck formation"
        }

        if (($collisions | Measure-Object -Maximum).Maximum -gt 0.0) {
            $reasons += "controlled-vehicle collision"
        }
        $rollbacks = @(
            $records | ForEach-Object {
                $followerRollback = if (
                    $_.PSObject.Properties.Name -contains
                    "training/follower_actor_rolled_back"
                ) {
                    [double]$_.'training/follower_actor_rolled_back'
                }
                else { 0.0 }
                $leaderRollback = if (
                    $_.PSObject.Properties.Name -contains
                    "training/leader_actor_rolled_back"
                ) {
                    [double]$_.'training/leader_actor_rolled_back'
                }
                else { 0.0 }
                [Math]::Max($followerRollback, $leaderRollback)
            }
        )
        $corrections = @(
            for ($index = 0; $index -lt $records.Count; $index++) {
                $candidatePending = [Math]::Max(
                    [double]$records[$index].'training/follower_actor_candidate_pending',
                    [double]$records[$index].'training/leader_actor_candidate_pending'
                )
                [Math]::Max($rollbacks[$index], $candidatePending)
            }
        )
        if (
            -not $actorReplayWarmingUp -and
            ($failures | Measure-Object -Sum).Sum -ge 2.0 -and
            ($corrections | Measure-Object -Sum).Sum -eq 0.0
        ) {
            $reasons += "repeated failures without an actor correction"
        }
        if (
            ($rollbacks | Measure-Object -Sum).Sum -ge 3.0 -and
            ($maxGaps | Measure-Object -Average).Average -gt 1.25 * $limit.max_gap
        ) {
            $reasons += "rejected actor candidates remain outside the safety envelope"
        }
        $maximumGap = ($maxGaps | Measure-Object -Maximum).Maximum
        if ($maximumGap -gt 150.0) {
            $reasons += "platoon separation exceeded the recoverable replay envelope"
        }
        elseif (
            -not $actorReplayWarmingUp -and
            $maximumGap -gt [Math]::Max(50.0, 2.5 * $limit.max_gap) -and
            ($corrections | Measure-Object -Sum).Sum -eq 0.0
        ) {
            $reasons += "severe platoon separation without an actor correction"
        }
        $unrecoveredMaxGaps = @(
            for ($index = 0; $index -lt $records.Count; $index++) {
                if ($corrections[$index] -eq 0.0) { $maxGaps[$index] }
            }
        )
        if (
            -not $actorReplayWarmingUp -and
            $unrecoveredMaxGaps.Count -ge 3 -and
            ($unrecoveredMaxGaps | Where-Object { $_ -gt 1.5 * $limit.max_gap }).Count -eq 3
        ) {
            $reasons += "persistent platoon separation"
        }
        if (($shields | Measure-Object -Average).Average -gt [Math]::Min(0.60, 2.0 * $limit.shield)) {
            $reasons += "persistent safety-shield intervention"
        }
        if (($strings | Measure-Object -Average).Average -gt 2.0 * $limit.string) {
            $reasons += "severe string-instability amplification"
        }
    }

    if ($reasons.Count -eq 0) {
        continue
    }

    $latestRecord = if ($records.Count -gt 0) { $records[-1] } else { $null }
    $alert = [ordered]@{
        stopped_at = (Get-Date).ToString("o")
        training_pid = $TrainingPid
        reasons = $reasons
        latest_record = $latestRecord
    }
    $alert | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $AlertPath -Encoding UTF8
    Stop-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    exit 2
}
