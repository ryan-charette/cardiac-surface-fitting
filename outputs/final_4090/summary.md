# RTX 4090 Benchmark Summary

## gpu_benchmark_4090.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=139.25345611572266, throughput=574492.0250562273, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.11703040190041065, throughput=683583057.9141102, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.7375696301460266, throughput=2992435.3009289578, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.32121919840574265, throughput=25502834.32826581, err=1.4901161193847656e-08

## gpu_benchmark_4090_chamfer_16384.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=140.0909881591797, throughput=571057.4323960029, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.10121759735047817, throughput=790376397.9201199, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 16384x16384: mean_ms=10.408696031570434, throughput=1574068.4472200912, err=0.0
- chamfer / triton_tiled_cuda / 16384x16384: mean_ms=0.8943311989307403, throughput=18319835.000264622, err=5.587935447692871e-09

## gpu_benchmark_4090_chamfer_2048.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=144.40200424194336, throughput=554008.9309699693, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.10243199802935124, throughput=781005950.6705756, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 2048x2048: mean_ms=0.28216320276260376, throughput=7258210.779961525, err=0.0
- chamfer / triton_tiled_cuda / 2048x2048: mean_ms=0.12298879995942116, throughput=16651922.782202246, err=0.0

## gpu_benchmark_4090_chamfer_4096.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=145.11821975708008, throughput=551274.6789060368, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.10081599801778793, throughput=793524852.9294412, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 4096x4096: mean_ms=0.6178751945495605, throughput=6629170.47994788, err=0.0
- chamfer / triton_tiled_cuda / 4096x4096: mean_ms=0.18722720071673393, throughput=21877163.06348594, err=1.4901161193847656e-08

## gpu_benchmark_4090_chamfer_8192.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=140.04607467651368, throughput=571240.5733954952, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.10062879845499992, throughput=795001045.707359, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.7337136387825014, throughput=2996656.2275514803, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.32095360159873965, throughput=25523938.535644617, err=1.4901161193847656e-08

## gpu_benchmark_4090_grid_1024x1024.csv

- tspline_eval / torch_framework_cuda / 1024x1024: mean_ms=145.07407760620117, throughput=7227866.048173852, err=2.069557168304925e-06
- tspline_eval / triton_degree2_cuda / 1024x1024: mean_ms=0.7228304028511048, throughput=1450652872.1869428, err=2.1996036663196605e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.748627173900604, throughput=2980396.933344238, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.32296000272035597, throughput=25365370.11084086, err=1.4901161193847656e-08

## gpu_benchmark_4090_grid_200x100.csv

- tspline_eval / torch_framework_cuda / 200x100: mean_ms=137.51907119750976, throughput=145434.37376242378, err=1.4545941606769475e-06
- tspline_eval / triton_degree2_cuda / 200x100: mean_ms=0.08339679837226868, throughput=239817359.7830879, err=1.618356415988842e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.7394960165023803, throughput=2990331.0501831067, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.32229760140180586, throughput=25417502.222696032, err=1.4901161193847656e-08

## gpu_benchmark_4090_grid_400x200.csv

- tspline_eval / torch_framework_cuda / 400x200: mean_ms=139.88274230957032, throughput=571907.5754388229, err=1.5934508050818863e-06
- tspline_eval / triton_degree2_cuda / 400x200: mean_ms=0.10076159797608852, throughput=793953267.9799758, err=1.8714061331692733e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.7395680069923403, throughput=2990252.470130742, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.3221088036894798, throughput=25432400.18952501, err=1.4901161193847656e-08

## gpu_benchmark_4090_grid_800x400.csv

- tspline_eval / torch_framework_cuda / 800x400: mean_ms=137.74626998901368, throughput=2323111.9073171453, err=1.8060987635459824e-06
- tspline_eval / triton_degree2_cuda / 800x400: mean_ms=0.25328320264816284, throughput=1263407903.3046415, err=2.056733707789249e-06
- chamfer / torch_cdist_cuda / 8192x8192: mean_ms=2.7389344215393066, throughput=2990944.191864229, err=0.0
- chamfer / triton_tiled_cuda / 8192x8192: mean_ms=0.32070719748735427, throughput=25543548.95737261, err=1.4901161193847656e-08
