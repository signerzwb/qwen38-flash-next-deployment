# 输出 JSON: {totals: {luid: bytes}, wsl: {luid: bytes}}
# luid = 0x{adapterLuid}, 与 nvidia-smi 的 GPU 顺序一致 (adapter index 升序)
# vmwp.exe 是 WSL2 的 Windows 侧 GPU 通道, 其上 Dedicated = WSL 内全部 CUDA 进程之和
$all = Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
$vmwpPid = (Get-Process -Name vmwp -ErrorAction SilentlyContinue | Select-Object -First 1).Id
$out = @{}
$wsl = @{}
foreach ($c in $all.CounterSamples) {
  $m = [regex]::Match($c.InstanceName, '^pid_(\d+)_luid_0x[0-9a-f]+_0x([0-9a-f]+)')
  if (-not $m.Success) { continue }
  $luid = 'luid_' + $m.Groups[2].Value
  $out[$luid] = [int64]($out[$luid] + $c.CookedValue)
  if ($vmwpPid -and $m.Groups[1].Value -eq [string]$vmwpPid) {
    $wsl[$luid] = [int64]($wsl[$luid] + $c.CookedValue)
  }
}
([pscustomobject]@{ totals = $out; wsl = $wsl } | ConvertTo-Json -Compress -Depth 5)
