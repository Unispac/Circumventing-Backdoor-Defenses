# Revisiting the Assumption of Latent Separability for Backdoor Defenses (ICLR 2023)

Official repostory for [Revisiting the Assumption of Latent Separability for Backdoor Defenses](https://openreview.net/forum?id=_wSHsgrVali) (ICLR 2023).

> Refer to https://github.com/vtu81/backdoor-toolbox for a more comprehensive backdoor research code repository, which includes our adaptive attacks, together with various other attacks and defenses.

## Attacks

Our proposed **adaptive** attacks:
- `adaptive_blend`: Adap-Blend attack with a single blending trigger
- `adaptive_patch`: Adap-Patch attack with `k` different patch triggers
- `adaptive_k_way`: Adap-K-Way attack, adaptive version of the [k-way attack](https://openreview.net/forum?id=rkgyS0VFvr)

Some other baselines include:
- `none`: no attack
- `badnet`: basic attack with badnet patch trigger
- `blend`: basic attack with a single blending trigger

See [poison_tool_box/](poison_tool_box/) for details.


## Defenses

We also include some backdoor defenses, including poison samples cleansers and other types of backdoor defenses. See [other_cleansers/](other_cleansers/) and [other_defenses/](other_defenses/) for details.

**Poison Cleansers**

- `SCAn`: https://arxiv.org/abs/1908.00686
- `AC`: activation clustering, https://arxiv.org/abs/1811.03728
- `SS`: spectral signature, https://arxiv.org/abs/1811.00636
- `SPECTRE`: https://arxiv.org/abs/2104.11315
- `Strip` (modified as a poison cleanser): http://arxiv.org/abs/1902.06531

**Other Defenses**

- `NC`: Neural Clenase, https://ieeexplore.ieee.org/document/8835365/
- `STRIP` (backdoor input filter): http://arxiv.org/abs/1902.06531
- `FP`: Fine-Pruning, http://arxiv.org/abs/1805.12185
- `ABL`: Anti-Backdoor Learning, https://arxiv.org/abs/2110.11571

## Visualization

Visualize the latent space of backdoor models. See [visualize.py](visualize.py).

- `tsne`: 2-dimensional T-SNE
- `pca`: 2-dimensional PCA
- `oracle`: fit the poison latent space with a SVM, see https://arxiv.org/abs/2205.13613

## Quick Start

Take launching and defending an Adaptive-Blend attack as an example:
```bash
# Create a clean set (for testing and some defenses)
python create_clean_set.py -dataset=cifar10

# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -no_aug

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003

# Cleanse poison train set with cleansers
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
## Except for 'CT', you need to train poisoned backdoor models first.
python other_cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003

# Retrain on cleansed set
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003

# Other defenses
## $DEFENSE = ['ABL', 'NC', 'STRIP', 'FP']
## Except for 'ABL', you need to train poisoned backdoor models first.
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003
```

**Notice**:
- `SPECTRE` is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.

Some other poisoning attacks we compare in our papers:
```bash
# No Poison
python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0
# BadNet
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.003
# Adaptive Patch
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006
# Adaptive K Way
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k_way -poison_rate=0.003 -cover_rate=0.003
```

You can also:
- train a vanilla model via
    ```bash
    python train_vanilla.py
    ```
- test a trained model via
    ```bash
    python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003
    # other options include: -no_aug, -cleanser=$CLEANSER, -model_path=$MODEL_PATH, see our code for details
    ```
- enforce a fixed running seed via `-seed=$SEED` option
- change dataset to GTSRB via `-dataset=gtsrb` option
- change model architectures in [config.py](config.py)
- configure hyperparamters of other defenses in [other_defense.py](other_defense.py)
- see more configurations in [config.py](config.py)
