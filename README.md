# ArtBench Generative Modeling

## Objetivo

Este projeto implementa e compara diferentes modelos generativos para síntese de imagens artísticas utilizando o dataset ArtBench-10. Os modelos implementados são:

- VAE (Variational Autoencoder)
- DCGAN (Deep Convolutional Generative Adversarial Network)
- Diffusion Model (DDPM - linear schedule)
- Advanced Diffusion Model (DDIM - cosine schedule)

## Dataset

ArtBench-10: 60.000 imagens de obras de arte (32x32 RGB) divididas em 10 categorias artísticas.

## Como Executar

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

### Treinamento

O script principal aceita dois argumentos:
- `--mode`: `subset` (20% do dataset) ou `full` (dataset completo)
- `--model`: `vae`, `gan`, `diffusion`, ou `diffusion_advanced`

Cada experimento treina o modelo com 10 seeds diferentes para robustez estatística e gera 5000 amostras por seed para calcular métricas FID e KID.

#### Exemplos:

```bash
# VAE com 20% do dataset (30 epochs)
python main.py --mode subset --model vae

# GAN com 20% do dataset (30 epochs)
python main.py --mode subset --model gan

# Diffusion básico com 20% do dataset (30 epochs)
python main.py --mode subset --model diffusion

# Diffusion avançado com 20% do dataset (30 epochs)
python main.py --mode subset --model diffusion_advanced

# VAE com dataset completo (50 epochs)
python main.py --mode full --model vae
```

### Scripts de Teste Rápido

Para testar individualmente cada modelo com 5 epochs:

```bash
python test-scripts/test_vae.py
python test-scripts/test_dcgan.py
python test-scripts/test_diffusion.py
python test-scripts/test_advanced_diffusion.py
```

## Resultados

Os resultados são salvos em `experiments/` com a seguinte estrutura:

```
experiments/
  <model>_<mode>_<timestamp>/
    experiment.log          # Métricas e configuração
    samples/
      seed_<X>_samples.png  # Grid de 36 amostras visuais
```

As métricas calculadas são:
- FID (Fréchet Inception Distance): menor é melhor
- KID (Kernel Inception Distance): menor é melhor

Resultados agregados (média e desvio padrão) são reportados no final do log.