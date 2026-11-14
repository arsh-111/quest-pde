# Stage 1: baseline and padding comparison (seed 0), 150 epochs each, on the MX450
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
& $py -m qpde.train --model cnn --epochs 150 --seed 0 --name cnn_s0
& $py -m qpde.train --model fno --epochs 150 --seed 0 --pad 0.0 --name fno_pad0_s0
& $py -m qpde.train --model fno --epochs 150 --seed 0 --pad 0.25 --name fno_pad25_s0
