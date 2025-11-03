"""
Installation test script to verify basic functionality.

Run this script after installation to check if everything is working.

Usage:
    python test_installation.py
"""

import sys


def test_imports():
    """Test if all required libraries are installed."""
    print("\n" + "="*60)
    print("1️⃣  Testing library imports...")
    print("="*60)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
        ('yfinance', 'yfinance'),
        ('seaborn', 'Seaborn'),
    ]
    
    optional_packages = [
        ('pyextremes', 'PyExtremes'),
        ('copulas', 'Copulas'),
        ('statsmodels', 'Statsmodels'),
    ]
    
    all_ok = True
    
    # Test required packages
    print("\nRequired packages:")
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name:<20} installed")
        except ImportError as e:
            print(f"   ❌ {display_name:<20} NOT FOUND")
            print(f"      Error: {e}")
            all_ok = False
    
    # Test optional packages
    print("\nOptional packages:")
    for module_name, display_name in optional_packages:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name:<20} instalado")
        except ImportError:
            print(f"   ⚠️  {display_name:<20} não instalado (opcional)")
    
    return all_ok


def test_project_structure():
    """Testa se a estrutura do projeto está correta."""
    print("\n" + "="*60)
    print("2️⃣  Testando estrutura do projeto...")
    print("="*60)
    
    import os
    
    required_files = [
        'README.md',
        'requirements.txt',
        'src/__init__.py',
        'src/utils.py',
        'src/phase1_gaussian_failure.py',
        'src/phase2_evt_engine.py',
        'src/phase3_risk_metrics.py',
        'src/phase4_copulas.py',
    ]
    
    all_ok = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} não encontrado")
            all_ok = False
    
    return all_ok


def test_basic_functionality():
    """Testa funcionalidade básica do projeto."""
    print("\n" + "="*60)
    print("3️⃣  Testando funcionalidade básica...")
    print("="*60)
    
    try:
        # Importar módulos do projeto
        print("\n   Importando módulos do projeto...")
        from src import utils
        from src.phase1_gaussian_failure import GaussianFailureAnalyzer
        from src.phase2_evt_engine import EVTEngine
        print("   ✅ Imports do projeto OK")
        
        # Testar funções básicas
        print("\n   Testando funções utilitárias...")
        import numpy as np
        
        # Criar dados de teste
        test_returns = np.random.normal(0, 0.01, 1000)
        
        # Testar cálculo de estatísticas
        stats_dict = {
            'mean': np.mean(test_returns),
            'std': np.std(test_returns),
            'kurtosis': 3.0
        }
        
        print("   ✅ Funções utilitárias OK")
        
        # Testar EVT Engine
        print("\n   Testando EVT Engine...")
        evt = EVTEngine(test_returns, ticker='TEST')
        evt.select_threshold(percentile=95)
        print("   ✅ EVT Engine OK")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Erro durante teste funcional:")
        print(f"      {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_download():
    """Testa download de dados do Yahoo Finance."""
    print("\n" + "="*60)
    print("4️⃣  Testando download de dados (yfinance)...")
    print("="*60)
    
    try:
        import yfinance as yf
        
        print("\n   Baixando dados de teste (SPY, últimos 5 dias)...")
        data = yf.download('SPY', period='5d', progress=False)
        
        if len(data) > 0:
            print(f"   ✅ Download bem-sucedido ({len(data)} dias)")
            print(f"      Data mais recente: {data.index[-1].strftime('%Y-%m-%d')}")
            return True
        else:
            print("   ⚠️  Download retornou dados vazios")
            print("      Verifique sua conexão com a internet")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Erro durante download:")
        print(f"      {e}")
        print("\n      Possíveis causas:")
        print("      • Sem conexão com internet")
        print("      • yfinance não instalado corretamente")
        print("      • Problemas temporários com Yahoo Finance")
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "="*80)
    print(" "*20 + "🧪 TESTE DE INSTALAÇÃO")
    print(" "*15 + "Motor de Risco Não-Gaussiano")
    print("="*80)
    
    print("\nEste script verifica se:")
    print("  • Todas as bibliotecas estão instaladas")
    print("  • A estrutura do projeto está correta")
    print("  • As funções básicas funcionam")
    print("  • O download de dados funciona")
    
    # Executar testes
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Estrutura", test_project_structure()))
    results.append(("Funcionalidade", test_basic_functionality()))
    results.append(("Download de dados", test_data_download()))
    
    # Resumo
    print("\n" + "="*80)
    print(" "*30 + "📊 RESUMO DOS TESTES")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"   {test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    # Conclusão
    if all_passed:
        print("\n" + "="*80)
        print(" "*25 + "✅ INSTALAÇÃO BEM-SUCEDIDA!")
        print("="*80)
        print("\n🎉 Tudo está funcionando corretamente!")
        print("\n📚 Próximos passos:")
        print("   1. Execute: python quickstart.py")
        print("   2. Ou execute: python demo_complete.py")
        print("   3. Leia o README.md para mais informações")
        print("\n" + "="*80 + "\n")
        return 0
    else:
        print("\n" + "="*80)
        print(" "*25 + "⚠️  ALGUNS TESTES FALHARAM")
        print("="*80)
        print("\n🔧 Ações recomendadas:")
        print("   1. Verifique se está no diretório correto do projeto")
        print("   2. Reinstale as dependências: pip install -r requirements.txt")
        print("   3. Verifique sua conexão com a internet")
        print("   4. Consulte INSTALL.md para mais detalhes")
        print("\n💡 Se o problema persistir, abra uma issue no GitHub")
        print("\n" + "="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

