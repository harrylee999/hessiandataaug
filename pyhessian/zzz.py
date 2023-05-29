import os




print(os.popen("python compute_hessian.py --model_dir ./logs/mixup+_last.pt --plot").read())
print(os.popen("python compute_hessian.py --model_dir ./logs/cutmix_last.pt --plot").read())
print(os.popen("python compute_hessian.py --model_dir ./logs/cutmix+_last.pt --plot").read())
print(os.popen("python compute_hessian.py --model_dir ./logs/noaug_last.pt --plot").read())
