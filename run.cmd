:base
python run.py -T resnet50 -X stabilityai/sd-vae-ft-mse
python test_vae.py -T resnet50 -X stabilityai/sd-vae-ft-mse -V 2
python vae_tmodel.py -T resnet50 -X stabilityai/sd-vae-ft-mse -V 2

:tiny
python run.py -T resnet18 -X madebyollin/taesd
python test_vae.py -T resnet18 -X madebyollin/taesd -V 0
python vae_tmodel.py -T resnet18 -X madebyollin/taesd -V 0
