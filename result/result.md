"task_name": "imdb",
"ood_datasets": "['imdb', 'multi30k']",
"model_class": "gpt2",
"do_train": false,
"num_train_epochs": 10,
"CUDA_VISIBLE_DEVICES": 0

INFO:utils.startup:On train data, Dispersion=0.3856295347213745; Compactness=4.265509605407715
INFO:utils.startup:
INFO:utils.startup:ood_multi30k_mmd: 0.06503850966691971
INFO:utils.startup:ood_multi30k_id-ood-seperability: 10.600486755371094
INFO:utils.startup:ood_multi30k_softmax: {'AUROC': 0.9994472353433836, 'AUPR-IN': 0.9991688491940289, 'AUPR-OUT': 0.9994337774535001, 'FPR95': 0.00011420740063956144}
INFO:utils.startup:ood_multi30k_energy: {'AUROC': 0.8611639834018578, 'AUPR-IN': 0.7742037833028533, 'AUPR-OUT': 0.8998406028388296, 'FPR95': 0.7503654636820466}
INFO:utils.startup:ood_multi30k_maha: {'AUROC': 0.2257078003654637, 'AUPR-IN': 0.23447911515081407, 'AUPR-OUT': 0.567639768434664, 'FPR95': 0.9929800517740216}
INFO:utils.startup:ood_multi30k_kNN: {'AUROC': 0.26577373404903304, 'AUPR-IN': 0.19147365428349167, 'AUPR-OUT': 0.6369051336178805, 'FPR95': 0.9321455763666819}
INFO:utils.startup:Done.


"task_name": "imdb",
"ood_datasets": "['imdb', 'multi30k']",
"model_class": "gpt2",
"do_train": true,
"num_train_epochs": 10,
"CUDA_VISIBLE_DEVICES": 0


INFO:utils.startup:{'dev_accuracy': 0.92276, 'dev_precision': array([0.91997139, 0.92558589]), 'dev_recall': array([0.92608, 0.91944]), 'dev_f1': array([0.92301559, 0.92250271]), 'dev_acc': 0.92276}
INFO:utils.startup:On train data, Dispersion=133.9880828857422; Compactness=7.882878303527832
INFO:utils.startup:
INFO:utils.startup:ood_multi30k_mmd: 0.6766829490661621
INFO:utils.startup:ood_multi30k_id-ood-seperability: 50.9677619934082
INFO:utils.startup:ood_multi30k_softmax: {'AUROC': 0.9847813687376276, 'AUPR-IN': 0.9822999880889751, 'AUPR-OUT': 0.9887695235107166, 'FPR95': 0.016308816811329375}
INFO:utils.startup:ood_multi30k_energy: {'AUROC': 0.9917145139333029, 'AUPR-IN': 0.9878611572362723, 'AUPR-OUT': 0.9939286735082381, 'FPR95': 0.0120831429876656}
INFO:utils.startup:ood_multi30k_maha: {'AUROC': 0.0464777496573778, 'AUPR-IN': 0.15894157948672463, 'AUPR-OUT': 0.5182115793431202, 'FPR95': 0.9999238617329069}
INFO:utils.startup:ood_multi30k_kNN: {'AUROC': 0.9885218493223693, 'AUPR-IN': 0.9851360623339905, 'AUPR-OUT': 0.9932187860296604, 'FPR95': 0.01167199634536318}



"task_name": "imdb",
"ood_datasets": "['yelp_polarity', 'imdb']",
"model_class": "gpt2",
"do_train": false,
"num_train_epochs": 10,
"CUDA_VISIBLE_DEVICES": 0

INFO:utils.startup:On train data, Dispersion=0.3856295347213745; Compactness=4.265509605407715
INFO:utils.startup:
INFO:utils.startup:ood_yelp_polarity_mmd: 0.007137594278901815
INFO:utils.startup:ood_yelp_polarity_id-ood-seperability: 2.9557442665100098
INFO:utils.startup:ood_yelp_polarity_softmax: {'AUROC': 0.6934951842105262, 'AUPR-IN': 0.6616243790262135, 'AUPR-OUT': 0.6954748392091202, 'FPR95': 0.7102894736842106}
INFO:utils.startup:ood_yelp_polarity_energy: {'AUROC': 0.4542604892105263, 'AUPR-IN': 0.5042687178746186, 'AUPR-OUT': 0.41523433892180506, 'FPR95': 0.9602894736842106}
INFO:utils.startup:ood_yelp_polarity_maha: {'AUROC': 0.6430571539473684, 'AUPR-IN': 0.6283533927232764, 'AUPR-OUT': 0.7154459902086863, 'FPR95': 0.5524736842105263}
INFO:utils.startup:ood_yelp_polarity_kNN: {'AUROC': 0.5064473018421052, 'AUPR-IN': 0.5272670000336593, 'AUPR-OUT': 0.5745474258104962, 'FPR95': 0.7179736842105263}


"task_name": "imdb",
"ood_datasets": "['yelp_polarity', 'imdb']",
"model_class": "gpt2",
"do_train": true,
"num_train_epochs": 10,
"CUDA_VISIBLE_DEVICES": 0

INFO:utils.startup:On train data, Dispersion=133.9880828857422; Compactness=7.882878303527832
INFO:utils.startup:
INFO:utils.startup:ood_yelp_polarity_mmd: 0.01482267864048481
INFO:utils.startup:ood_yelp_polarity_id-ood-seperability: 7.215607643127441
INFO:utils.startup:ood_yelp_polarity_softmax: {'AUROC': 0.692387762368421, 'AUPR-IN': 0.7554654935509402, 'AUPR-OUT': 0.5852503861366624, 'FPR95': 0.8879736842105264}
INFO:utils.startup:ood_yelp_polarity_energy: {'AUROC': 0.6785897344736842, 'AUPR-IN': 0.745158215807482, 'AUPR-OUT': 0.5803988018166446, 'FPR95': 0.8873947368421052}
INFO:utils.startup:ood_yelp_polarity_maha: {'AUROC': 0.580382517631579, 'AUPR-IN': 0.565576004852089, 'AUPR-OUT': 0.6635075320558121, 'FPR95': 0.6183684210526316}
INFO:utils.startup:ood_yelp_polarity_kNN: {'AUROC': 0.7610857421052631, 'AUPR-IN': 0.8351295454102698, 'AUPR-OUT': 0.6493689620278839, 'FPR95': 0.8520526315789474}




ood_yelp_polarity_mmd:tensor(0.0230, device='cuda:0')
ood_yelp_polarity_id-ood-seperability:7.12265
ood_yelp_polarity_softmax:{'AUROC': 0.5097504260526317, 'AUPR-IN': 0.5765972891187162, 'AUPR-OUT': 0.4386982235227751, 'FPR95': 0.9471052631578948}
ood_yelp_polarity_energy:{'AUROC': 0.5010663876315791, 'AUPR-IN': 0.5739004014948205, 'AUPR-OUT': 0.43036937884775833, 'FPR95': 0.9526315789473684}
ood_yelp_polarity_maha:{'AUROC': 0.9583061907894737, 'AUPR-IN': 0.9713994625054516, 'AUPR-OUT': 0.9320247575727311, 'FPR95': 0.22223684210526315}
ood_yelp_polarity_kNN:{'AUROC': 0.941457642894737, 'AUPR-IN': 0.9623024221336243, 'AUPR-OUT': 0.9006891524315324, 'FPR95': 0.3948421052631579}

"task_name": "imdb",
"ood_datasets": "['yelp_polarity', 'imdb']",
"model_class": "roberta",
"do_train": true,
"use_adapter": true,
"num_train_epochs": 6,
"CUDA_VISIBLE_DEVICES": 0

ood_yelp_polarity_mmd:tensor(0.0230, device='cuda:0')
ood_yelp_polarity_id-ood-seperability:7.122654
ood_yelp_polarity_softmax:{'AUROC': 0.5127994565789473, 'AUPR-IN': 0.5814283939701591, 'AUPR-OUT': 0.4388354603558992, 'FPR95': 0.9508684210526316}
ood_yelp_polarity_energy:{'AUROC': 0.5008578336842106, 'AUPR-IN': 0.572897902771742, 'AUPR-OUT': 0.4292035807929634, 'FPR95': 0.9546842105263158}
ood_yelp_polarity_maha:{'AUROC': 0.9583061792105263, 'AUPR-IN': 0.9713994485618439, 'AUPR-OUT': 0.9320247537603492, 'FPR95': 0.22221052631578947}
ood_yelp_polarity_kNN:{'AUROC': 0.941457642894737, 'AUPR-IN': 0.9623024221336243, 'AUPR-OUT': 0.9006891524315324, 'FPR95': 0.3948421052631579}
