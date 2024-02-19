# LOSS

label_smoothing_CE : only change the function "def cross_entropy"

action_head_loss_fixed : change get_masked_classify_loss_for_multi_gpu function in line 289,
                         fixed each action_head loss like di-star loss.
