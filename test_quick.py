"""
Script de teste rápido - Valida que tudo funciona antes do treino completo.
Roda 1 época com 1 seed apenas para verificar:
- Carregamento de dados
- Treino de cada modelo
- Geração de amostras
- Cálculo de métricas
"""

import numpy as np
import random
import torch
from models.conv_vae_model import ConvVAE
from models.dcgan_model import Generator, Discriminator, weights_init
from models.diffusion_model import UNet
from data import get_train_loader_from_csv, get_test_images_tensor, save_sample_grid
from training.conv_vae_trainer import ConvVAETrainer
from training.dcgan_trainer import DCGANTrainer
from training.diffusion_trainer import DiffusionTrainer
from utils.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SEED = 42
NUM_TEST_SAMPLES = 100  # Apenas 100 amostras para teste rápido

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def test_model(model_name):
    print(f"\n{'='*60}")
    print(f"TESTE RÁPIDO: {model_name.upper()}")
    print(f"{'='*60}\n")
    
    set_seed(TEST_SEED)
    
    # 1. Carregar dados
    print("1. Carregando dados...")
    train_loader = get_train_loader_from_csv()
    print(f"   ✓ Train loader: {len(train_loader)} batches")
    
    # 2. Construir modelo
    print(f"\n2. Construindo modelo {model_name}...")
    if model_name == "vae":
        model = ConvVAE(latent_dim=20).to(DEVICE)
        trainer = ConvVAETrainer(model, DEVICE, lr=1e-3, beta=1.0)
    elif model_name == "gan":
        generator = Generator(z_dim=100).to(DEVICE)
        discriminator = Discriminator().to(DEVICE)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        trainer = DCGANTrainer(generator, discriminator, DEVICE, lr=2e-4)
        model = (generator, discriminator)
    elif model_name == "diffusion":
        model = UNet(image_channels=3, base_channels=64, time_emb_dim=128).to(DEVICE)
        trainer = DiffusionTrainer(model, DEVICE, lr=1e-4, timesteps=1000, use_ema=True)
    
    total_params = sum(p.numel() for p in (model.parameters() if model_name == "vae" else 
                      (list(model[0].parameters()) + list(model[1].parameters()) if model_name == "gan" else model.parameters())))
    print(f"   ✓ Modelo criado: {total_params:,} parâmetros")
    
    # 3. Treinar 1 época
    print(f"\n3. Treinando 1 época...")
    trainer.fit(train_loader, val_loader=None, num_epochs=1)
    print(f"   ✓ Treino concluído")
    
    # 4. Gerar amostras
    print(f"\n4. Gerando {NUM_TEST_SAMPLES} amostras...")
    with torch.no_grad():
        if model_name == "vae":
            samples = model.sample(NUM_TEST_SAMPLES, DEVICE)
        else:
            samples = trainer.sample(NUM_TEST_SAMPLES)
    print(f"   ✓ Amostras geradas: shape={samples.shape}")
    
    # 5. Salvar grid
    print(f"\n5. Salvando grid de amostras...")
    save_sample_grid(samples[:36], f"outputs/test_{model_name}_grid.png", nrow=6)
    print(f"   ✓ Grid salvo em outputs/test_{model_name}_grid.png")
    
    # 6. Calcular métricas (com menos amostras para ser rápido)
    print(f"\n6. Calculando métricas FID/KID (teste com {NUM_TEST_SAMPLES} amostras)...")
    real = get_test_images_tensor(device=DEVICE)
    real = real[:NUM_TEST_SAMPLES]  # Usar menos amostras reais também
    
    fid, kid_mean, kid_std = compute_metrics(
        real,
        samples,
        use_cuda=torch.cuda.is_available()
    )
    print(f"   ✓ FID: {fid:.4f}")
    print(f"   ✓ KID: {kid_mean:.6f} ± {kid_std:.6f}")
    
    print(f"\n{'='*60}")
    print(f"✅ TESTE {model_name.upper()} PASSOU!")
    print(f"{'='*60}\n")
    
    return True

def main():
    print(f"\n{'#'*60}")
    print(f"# TESTE RÁPIDO DE VALIDAÇÃO")
    print(f"# Device: {DEVICE}")
    print(f"# Seed: {TEST_SEED}")
    print(f"# Épocas: 1 (teste)")
    print(f"# Amostras: {NUM_TEST_SAMPLES} (teste)")
    print(f"{'#'*60}\n")
    
    models_to_test = ["vae", "gan", "diffusion"]
    
    results = {}
    for model_name in models_to_test:
        try:
            success = test_model(model_name)
            results[model_name] = "✅ PASSOU"
        except Exception as e:
            print(f"\n❌ ERRO no {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = f"❌ FALHOU: {str(e)}"
    
    # Sumário final
    print(f"\n{'='*60}")
    print("SUMÁRIO DOS TESTES")
    print(f"{'='*60}")
    for model_name, status in results.items():
        print(f"{model_name.upper():12s}: {status}")
    print(f"{'='*60}\n")
    
    all_passed = all("✅" in status for status in results.values())
    
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\nVocê pode agora rodar o protocolo completo:")
        print("  python main.py --mode subset --model vae")
        print("  python main.py --mode subset --model gan")
        print("  python main.py --mode subset --model diffusion")
    else:
        print("⚠️  ALGUNS TESTES FALHARAM!")
        print("Corrija os erros antes de rodar o protocolo completo.")

if __name__ == "__main__":
    main()
