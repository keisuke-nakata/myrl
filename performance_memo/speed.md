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
#### setting 1 (20180802_065234)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 369.88 fps くらい
- limit = 300_000 (disk usage = 88% (もともと 55% くらいは使っているので、33% (=5GB) くらいの消費量)。memory usage は 2GB くらい (謎))
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 44.21 くらい
- actor fps = 319.87 fps くらい

load average は 4 くらい
GPU 使用率は 32% くらい


## n1-standard-8（vCPU x 8、メモリ 30 GB）, 1 x NVIDIA Tesla K80 (preemptive)
learner の sample を prefetch かけるようにした (n_prefetches = 2).
CPU の使用率的には, 5 コアくらいで十分そうな感じ
#### setting 1 (____) ※一旦コードが動くことを確認したあとすぐにインスタンスを潰したので、結果はなし
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 5_000
- fps (warmup) = 369.88 fps くらい
- limit = 300_000 (disk usage = 88% (もともと 55% くらいは使っているので、33% (=5GB) くらいの消費量)。memory usage は 2GB くらい (謎))
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 55-72 くらい。けっこうブレる
- actor fps = 271.49 - 314.04 fps くらい (ちょっと遅くなった？ 一番速いときは同じくらいの速度なので、sampler 側の prefetch でロックをかけているせいか？)

load average は 3.2 くらい
GPU 使用率は 40% くらいまであがった


## custom（vCPU x 6、メモリ 16 GB）, 1 x NVIDIA Tesla K80 (preemptive)
CPU の数を 6個にする (偶数個しか選べない)
learner の sample を prefetch の数を 2 から 3 に増やす
リプレイのためには disk も必要そう (謎) なので、disk も 15GB -> 40GB に増やす (メモリはそんなにいらなそう？RawArray の実装どうなってんだ？memory-mapped file なのか？)
#### setting 1 (20180802_220440)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- fps (warmup) = 380 fps くらい
- limit = 400_000 (標準ディスクではなく SSD にすると、allocate が全く終わらない。謎 → SSD のせいではなく、メモリが足りずに詰んでいただけっぽい)
- Memory length at episode ____: ____
- batch_size = 32

- batches per seconds = 58.72 くらい。けっこうブレる
- actor fps = 258.25 fps くらい (明らか遅い。prefetch が大きすぎたか。)
- load average : 3.5 くらい
- GPU 使用率 : 35-40 % くらい
- disk usage = 15GB (もともと 7.4GB は使っているので、 7.6GB くらいの消費量)
- memory usage = 最初の方: 1GB くらい  warmup後：1.8GBくらい


## custom（vCPU x 6、メモリ 16 GB）, 1 x NVIDIA Tesla K80 (preemptive)
メモリが全然いらなさそうなので、12GB まで減らす・・・と思ったけど、値段が全然変わらないので 16GB キープ (3$/monthしか安くならない)
learner の sample を prefetch の数を 1 に減らす
#### setting 1 (____) ※一旦コードが動くことを確認したあとすぐにインスタンスを潰したので、結果はなし
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- limit = 600_000
- Memory length at episode ____: ____
- batch_size = 32

- fps (warmup) = 375.88 fps くらい
- batches per seconds = 50-55 くらい。けっこうブレる
- actor fps = 222.27 fps くらい (n_prefetches = 1 でも逆に効率が悪いのか？また =2 で実験する必要がある)
- load average : 3.2-3.5 くらい
- GPU 使用率 : 33-45 % くらい
- disk usage = 18 GB (もともと 7.4GB は使っているので、 10.6 GB くらいの消費量。 memory usage と同じくらいだ！)
- memory usage = 68.2% (=10.7GB)


## custom（vCPU x 6、メモリ 16 GB）, 1 x NVIDIA Tesla K80 (preemptive)
learner の sample を prefetch の数を 2 に戻す
#### setting 1 (____)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- limit = 600_000
- Memory length at episode ____: ____
- batch_size = 32

- fps (warmup) = 385.05 fps くらい
- batches per seconds = 45 - 60 くらい。かなりブレる
- actor fps = 254.09 fps くらい
- load average : 4.2 - 4.4 くらい
- GPU 使用率 : 35 - 45 % くらい
- disk usage = 18 GB (もともと 7.4GB は使っているので、 10.6 GB くらいの消費量。 ちゃんと memory usage と同じくらいになっている)
- memory usage = 68.2% (=10.7GB)


# BreakoutNoFrameskip-v4
- 1 episode あたりの step 数： 700 - 1,500 ほど (うまくいくほど伸びる)
- 1 episode あたりのダンプサイズ：____ MB くらい
  - なおす 50_000_000 steps ~ 50_000 episodes なので、 全エピソードをダンプすると、greedy ぶんもあわせて 2 * ____ MB * 50_000 = 1,600 GB
  - なおす 100 episode ごとにダンプで、 16 GB 必要となる


## custom（vCPU x 6、メモリ 16 GB）, 1 x NVIDIA Tesla K80 (preemptive), standard disk 40GB
#### setting 1 (____)
- actor: GPU
- greedy actor: GPU
- learner: GPU
- warmup = 50_000
- limit = 600_000
- Memory length at episode ____: ____
- batch_size = 32
- learner prefetch = 2

- fps (warmup) = 385.62 fps くらい
- batches per seconds = 45 - 60 くらい。かなりブレる
- actor fps = 254.09 fps くらい
- load average : 2.6 - 2.8 くらい (Alien よりも結構余裕がある。なぜ？)
- GPU 使用率 : 35 - 45 % くらい
- disk usage = 18 GB (もともと 7.4GB は使っているので、 10.6 GB くらいの消費量。 ちゃんと memory usage と同じくらいになっている)
- memory usage = 68.2% (=10.7GB)



やること
- SSD
- CPU もっと余裕もたせると actor 速くなるのでは?
