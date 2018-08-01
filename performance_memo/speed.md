# AlienNoFrameskip-v4
- 1 episode あたりの step 数： 2,000 - 3,000 ほど
- 1 episode あたりのダンプサイズ：30 - 50 MB くらい
  - 50_000_000 steps ~ 20_000 episodes なので、 全エピソードをダンプすると、greedy ぶんもあわせて 2 * 40 MB * 20_000 = 1,600 GB
  - 100 episode ごとにダンプで、 16 GB 必要となる
## n1-highmem-4（vCPU x 4、メモリ 26 GB）, 1 x NVIDIA Tesla K80 (preemptive)

### VanillaDQNAgent
#### setting 1 (20180715_040648)
- warmup = 50_000
- fps (warmup) = 554.03 くらい
- limit = 400_000
- batch_size = 32

- 151.72 fps くらい (VanillaDQNAgent なので actor と learner はくっついてる)

### AsyncDQNAgent
#### setting 1 (20180724_005110)
- actor: GPU
- greedy actor: CPU
- learner: GPU
- warmup = 100
- fps (warmup) = 不明
- limit = 400_000 (episode 131, total_steps 346,973 で止まった。MemoryError ではなくインスタンスの停止か？しかし stackdriver 上ではメモリ使い果たしていた)
- Memory length at episode 100: 66124
- batch_size = 32

- batches per seconds = 26.20 くらい (最初の方)  18.52 くらい (total_updates=40,000 のあたり)
- actor fps = 182.52 くらい

### AsyncDQNAgent (memory に格納する state を uint8 にしたバージョン)
#### setting 1 (20180725_014722)
- actor: GPU
- greedy actor: CPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 304.2 くらい
- limit = 400_000 (____)
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 32.8 くらい (total_updates=15,000-50,000)  22 くらい (total_updates=25,000以降)
- actor fps = 187.7 くらい (最初の20エピソード)  200 くらい (100-500エピソード)

### AsyncDQNAgent (Shared memory の multiprocessing バージョン)
#### setting 1 MacBookPro (20180729_040706)
- actor: CPU
- greedy actor: CPU
- learner: CPU
- warmup = 5_000
- fps (warmup) = 短すぎて測定不能 (参考：512.45 くらい)
- limit = 400_000 (____)  途中で潰したので略
- Memory length at episode ____: ____  limit は短くしたので測定不能
- batch_size = 32

- batches per seconds = 6.00 くらい
- actor fps = 300 くらい

### AsyncDQNAgent (Shared memory の multiprocessing バージョン)
limit = 400_000, disk = 15GB だと、shared memory の準備の時点で溢れて死亡
#### setting 1 ()
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 384.02 fps くらい
- limit = 300_000 (disk usage = 88%)
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 30.30 くらい
- actor fps = 230.77 fps くらい

どうやら CPU を使い切っている？ようす (load average が 4 近くなっている) なので、CPU の数を増やしたほうが良さそう

### AsyncDQNAgent (Shared memory の multiprocessing バージョン, actor を CPU へ)
#### setting 2 (20180731_031611)
- actor: CPU
- greedy actor: CPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 351.07 fps くらい
- limit = 300_000
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 24.89 くらい
- actor fps = 173.68 fps くらい

load average が 5 近い。actor を CPU にしたら、さらに CPU 足りなくなった


## n1-standard-8（vCPU x 8、メモリ 30 GB）, 1 x NVIDIA Tesla K80 (preemptive)
actor の memory push を step ごとではなく episode ごとにした  
また、Array/Value の代わりに RawArray/RawValue を利用することにした
※CPUのコア数を倍に増やしているので注意 (メモリも若干増えてる)  ← CPU の数は5個がちょうどよさそう
#### setting 1 (____)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 369.88 fps くらい
- limit = 300_000 (disk usage = 88% (もともと 55% くらいは使っているので、33% (=5GB) くらいの消費量))
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 44.21 くらい
- actor fps = 319.87 fps くらい

load average は 4 くらい


## custom（vCPU x 5、メモリ 30 GB）, 1 x NVIDIA Tesla K80 (preemptive)
リプレイのためには disk も必要そう (謎) なので、disk も 15GB -> 40GB に増やす
#### setting 1 (____)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = ____ fps くらい
- limit = ____ (disk usage = ____% (もともと 55% くらいは使っているので、33% (=5GB) くらいの消費量))
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = ____くらい
- actor fps = ____ fps くらい
