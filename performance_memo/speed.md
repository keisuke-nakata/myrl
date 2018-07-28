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
