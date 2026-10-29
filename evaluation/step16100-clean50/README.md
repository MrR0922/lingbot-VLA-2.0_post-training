# LingBot step16100 RoboTwin 50任务 clean 评测记录

已核验50项、每项100个有效回合，共5000回合；成功479次，合计/任务均值成功率均为9.58%。

## 协议和提交状态

这是在已披露修改协议下的实测结果，不是官方未修改协议的认证成绩。材料包尚未向赛事平台提交；未取得赛事专用提交格式、参赛身份或平台验收信息。模型权重单独归档。

权重固定为 global_step_16100/hf_ckpt；expert_check=false，ROBOTWIN_DISABLE_CUROBO=1，seen指令，无视频。训练未恢复。

结果来自36项valid7、3项resume14和11项repair11；没有混入valid2-valid6。每任务只采用唯一完整100回合结果，没有按成功率选优；中断尝试不参与分母，完整尝试日志均保留。

repair11非专家初始化遇UnStableError时记录并跳过无效场景，直到完成100个有效回合。因此各任务实际种子集合可能不同，不能等同固定100000–100099种子或专家可解性筛选；results.json列出有效和排除种子。

repair11取消非专家重复setup_demo；三个任务将play_once中的arm_tag/origin_z和指令info初始化到setup_demo，不改变成功判定公式。官方指令生成为空时回退原始任务模板。未证明模板回退和原始官方指令完全等价。

websocket接收180秒超时、ping20/60；后续启动客户端另有300秒无进展堆栈转储退出保护。摄像头get_picture阻塞曾导致中断重试，相关堆栈见日志。超时和初始化错误没有记为模型失败回合。

原39项与repair11存在实现和种子处理差异，合并成绩须连同此协议说明使用，不可省略差异直接宣传为标准官方分数。

## 逐任务结果

|任务|成功/有效回合|成功率|排除场景数（最终尝试）|
|---|---:|---:|---:|
|lift_pot|0/100|0%|0|
|hanging_mug|0/100|0%|0|
|stack_bowls_three|0/100|0%|0|
|scan_object|0/100|0%|0|
|handover_block|0/100|0%|0|
|click_bell|46/100|46%|0|
|put_object_cabinet|0/100|0%|0|
|open_microwave|66/100|66%|0|
|stack_blocks_three|0/100|0%|0|
|place_shoe|0/100|0%|0|
|adjust_bottle|0/100|0%|36|
|beat_block_hammer|4/100|4%|0|
|blocks_ranking_rgb|1/100|1%|0|
|blocks_ranking_size|0/100|0%|0|
|click_alarmclock|53/100|53%|0|
|dump_bin_bigbin|0/100|0%|94|
|grab_roller|0/100|0%|0|
|handover_mic|0/100|0%|0|
|move_can_pot|19/100|19%|1|
|move_pillbottle_pad|0/100|0%|0|
|move_playingcard_away|15/100|15%|0|
|place_cans_plasticbox|0/100|0%|0|
|place_container_plate|1/100|1%|0|
|place_dual_shoes|0/100|0%|7|
|place_empty_cup|0/100|0%|0|
|place_fan|0/100|0%|0|
|place_mouse_pad|0/100|0%|0|
|place_object_basket|0/100|0%|0|
|place_object_scale|0/100|0%|0|
|place_object_stand|1/100|1%|0|
|place_phone_stand|1/100|1%|0|
|move_stapler_pad|0/100|0%|0|
|open_laptop|15/100|15%|0|
|pick_diverse_bottles|0/100|0%|117|
|pick_dual_bottles|0/100|0%|0|
|place_a2b_left|7/100|7%|0|
|place_a2b_right|6/100|6%|0|
|place_bread_basket|2/100|2%|0|
|place_bread_skillet|0/100|0%|1|
|place_burger_fries|0/100|0%|0|
|place_can_basket|0/100|0%|0|
|press_stapler|96/100|96%|0|
|rotate_qrcode|3/100|3%|0|
|shake_bottle_horizontally|41/100|41%|58|
|shake_bottle|45/100|45%|58|
|stack_blocks_two|0/100|0%|0|
|stack_bowls_two|0/100|0%|0|
|stamp_seal|1/100|1%|0|
|turn_switch|54/100|54%|0|
|put_bottles_dustbin|2/100|2%|0|
