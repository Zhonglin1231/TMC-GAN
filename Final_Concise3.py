import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds - 完全复制myDefines2的设置
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import custom modules
from myDefines3 import preprocess_data, EnhancedUnifiedDataset, EnhancedGenerator, EnhancedDiscriminator, train_gan, predict_unified, calculate_metrics

# ============================================
# 更新参数设置以匹配myDefines2的改进
# ============================================
PARAMS = {
    'n_heads': 2,
    'WINDOW_SIZE': 50,
    'macro_weight_target': 0.3,
    'LAMBDA_SPECTRAL': 0.0,
    # Fixed parameters
    'BATCH_SIZE': 16,
    'NUM_EPOCHS': 30,
    'HIDDEN_DIM': 96,
    'N_LAYERS': 2,
    'DROPOUT': 0.3,
    'USE_RETURNS': True,
    'TEST_SPLIT': 0.2,
    'LAMBDA_REG': 0.5,
    'LAMBDA_VARIANCE': 10,
    'LAMBDA_DIRECTIONAL': 0.5
}

# ============================================
# 更新实验函数以匹配myDefines2的改进
# ============================================
def Run_Experiment(PARAMS, myDataset="SP500"):
    """完全匹配myDefines2逻辑的实验函数"""
    print(f"\n{'='*60}")
    print(f"运行单次实验 - 使用增强版模型架构")
    print(f"数据集: {myDataset}")
    print(f"{'='*60}")
    
    try:
        # Data preprocessing - 使用myDefines2的改进版预处理
        print("🔧 数据预处理中...")
        data = preprocess_data(
            f"Datasets/merged{myDataset}.csv",
            window_size=PARAMS['WINDOW_SIZE'],
            use_returns=PARAMS['USE_RETURNS'],
            test_split=PARAMS['TEST_SPLIT'],
            dataset=myDataset
        )
        
        (X_stock_train, X_macro_train, y_train, prev_train, freq_train, cyc_train,
         X_stock_test, X_macro_test, y_test, prev_test, freq_test, cyc_test,
         scalers, df, date_correspondences, base_prices_train, base_prices_test) = data
        
        print(f"📊 数据加载完成:")
        print(f"  训练样本: {len(X_stock_train)}")
        print(f"  测试样本: {len(X_stock_test)}")
        print(f"  股票特征维度: {X_stock_train.shape[2]}")
        print(f"  宏观特征维度: {X_macro_train.shape[1] * X_macro_train.shape[2]}")
        print(f"  频域特征维度: {freq_train.shape[1]}")
        print(f"  周期特征维度: {cyc_train.shape[1]}")
        
        # Create dataset and dataloader - 使用增强版数据集
        train_dataset = EnhancedUnifiedDataset(
            X_stock_train, X_macro_train, y_train, freq_train, cyc_train, prev_train
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=PARAMS['BATCH_SIZE'], 
            shuffle=True,
            drop_last=True  # 确保批次大小一致
        )
        
        # Model dimensions - 计算正确的维度
        dims = {
            'stock_input_dim': X_stock_train.shape[2],
            'macro_input_dim': X_macro_train.shape[1] * X_macro_train.shape[2],
            'freq_dim': freq_train.shape[1],
            'cyc_dim': cyc_train.shape[1],
            'output_dim': 1
        }
        
        print(f"🏗️ 模型维度:")
        for key, value in dims.items():
            print(f"  {key}: {value}")
        
        # Initialize models - 使用增强版模型架构
        print("🤖 初始化生成器...")
        generator = EnhancedGenerator(
            stock_input_dim=dims['stock_input_dim'], 
            macro_input_dim=dims['macro_input_dim'], 
            freq_dim=dims['freq_dim'], 
            cyclical_dim=dims['cyc_dim'],
            hidden_dim=PARAMS['HIDDEN_DIM'], 
            output_dim=dims['output_dim'],
            n_heads=PARAMS['n_heads'], 
            n_layers=PARAMS['N_LAYERS'], 
            dropout=PARAMS['DROPOUT']
        ).to(device)
        
        print("🛡️ 初始化判别器...")
        discriminator = EnhancedDiscriminator(
            stock_input_dim=dims['stock_input_dim'],
            macro_input_dim=dims['macro_input_dim'], 
            freq_dim=dims['freq_dim'], 
            cyclical_dim=dims['cyc_dim'],
            hidden_dim=PARAMS['HIDDEN_DIM'], 
            n_heads=PARAMS['n_heads'], 
            n_layers=PARAMS['N_LAYERS'], 
            dropout=PARAMS['DROPOUT']
        ).to(device)
        
        # 权重调整逻辑 - 改进版
        param_name = 'macro_weight_target'  # 明确指定当前调优参数
        
        print(f"🔧 权重调整逻辑:")
        print(f"  当前参数: {param_name}")
        print(f"  目标宏观权重: {PARAMS['macro_weight_target']}")
        print(f"  基线宏观权重: {PARAMS['macro_weight_target']}")
        
        # 检查是否需要调整权重
        if param_name == 'macro_weight_target' or PARAMS['macro_weight_target'] != PARAMS['macro_weight_target']:
            print("✅ 执行权重调整")
            with torch.no_grad():
                new_weights = torch.tensor([
                    2.0,  # stock - 主要权重
                    PARAMS['macro_weight_target'],  # macro - 可调权重
                    0.2,  # freq - 固定权重
                    0.1   # cyclical - 固定权重
                ]).to(device)
                generator.fusion_weights.data = new_weights
            print(f"  调整后权重: {generator.fusion_weights.data}")
        else:
            print("❌ 使用默认权重")
            print(f"  默认权重: {generator.fusion_weights.data}")
        
        # 模型参数统计
        total_PARAMS = sum(p.numel() for p in generator.parameters())
        trainable_PARAMS = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"📊 生成器参数: 总计={total_PARAMS:,}, 可训练={trainable_PARAMS:,}")
        
        total_PARAMS_d = sum(p.numel() for p in discriminator.parameters())
        trainable_PARAMS_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"📊 判别器参数: 总计={total_PARAMS_d:,}, 可训练={trainable_PARAMS_d:,}")
        
        # Training - 使用增强版训练函数
        print("🚀 开始训练...")

        time_start = time.time()  # 记录开始时间

        losses, fusion_weights_history = train_gan(
            generator=generator, 
            discriminator=discriminator, 
            dataloader=train_loader, 
            num_epochs=PARAMS['NUM_EPOCHS'],
            g_lr=0.0001,  # 使用myDefines2的默认学习率
            d_lr=0.0001,
            lambda_reg=PARAMS['LAMBDA_REG'], 
            lambda_variance=PARAMS['LAMBDA_VARIANCE'],
            lambda_spectral=PARAMS['LAMBDA_SPECTRAL'], 
            lambda_directional=PARAMS['LAMBDA_DIRECTIONAL'],
            clip_value=1.0,  # 梯度裁剪
            use_directional_loss=PARAMS['USE_RETURNS']
        )
        
        time_end = time.time()  # 记录结束时间
        elapsed_time = time_end - time_start
        print(f"⏱️ 训练时间: {elapsed_time:.2f} 秒")


        print("✅ 训练完成！")
        
        # 显示训练过程中的权重变化
        if fusion_weights_history:
            final_weights = fusion_weights_history[-1]
            print(f"🎯 最终融合权重: 股票={final_weights[0]:.3f}, 宏观={final_weights[1]:.3f}, "
                  f"频域={final_weights[2]:.3f}, 周期={final_weights[3]:.3f}")
        
        # Evaluation - 使用改进的评估方法
        print("📈 模型评估中...")
        generator.eval()
        
        predictions = []
        predicted_prices = []
        actual_prices = []
        actual_returns = []
        prev_values_for_direction = []
        
        # 获取实际价格和收益率
        test_start_idx = int(len(date_correspondences) * (1 - PARAMS['TEST_SPLIT']))
        
        for i in range(len(X_stock_test)):
            if test_start_idx + i < len(date_correspondences):
                actual_prices.append(date_correspondences[test_start_idx + i]['target_close_price'])
        
        # 预测
        with torch.no_grad():
            time_start = time.time()  # 记录预测开始时间
            for i in range(len(X_stock_test)):
                if PARAMS['USE_RETURNS']:

                    pred_price, fusion_weights, pred_return = predict_unified(
                        generator, X_stock_test[i], X_macro_test[i], freq_test[i], cyc_test[i],
                        scalers, last_price=base_prices_test[i], use_returns=PARAMS['USE_RETURNS']
                    )
                    predictions.append(pred_return)
                    predicted_prices.append(pred_price)
                    prev_values_for_direction.append(base_prices_test[i])
                else:
                    pred_price, fusion_weights, _ = predict_unified(
                        generator, X_stock_test[i], X_macro_test[i], freq_test[i], cyc_test[i],
                        scalers, use_returns=PARAMS['USE_RETURNS']
                    )
                    predictions.append(pred_price)
                    predicted_prices.append(pred_price)
                    if i > 0:
                        prev_values_for_direction.append(predicted_prices[i-1])
        # 转换为数组
        predictions = np.array(predictions)
        predicted_prices = np.array(predicted_prices)
        actual_prices = np.array(actual_prices[:len(predicted_prices)])
        prev_values_for_direction = np.array(prev_values_for_direction)
        


        # 获取实际收益率
        if PARAMS['USE_RETURNS']:
            if scalers['target_scaler']:
                actual_returns = scalers['target_scaler'].inverse_transform(y_test)
            else:
                actual_returns = y_test
            actual_returns = actual_returns.flatten()[:len(predictions)]
        else:
            actual_returns = y_test.flatten()[:len(predictions)]
        
        time_end = time.time()  # 记录预测结束时间
        elapsed_time = time_end - time_start
        print(f"⏱️ 预测时间: {elapsed_time:.2f} 秒")

        # 计算指标 - 使用改进的计算方法
        print("📊 计算性能指标...")
        
        # 价格指标（使用前一天价格作为参考）
        price_metrics = calculate_metrics(
            predicted_prices, actual_prices, 
            prev_values=prev_values_for_direction if len(prev_values_for_direction) == len(predicted_prices) else None,
            prefix="Price"
        )
        
        # 收益率指标
        if PARAMS['USE_RETURNS']:
            return_metrics = calculate_metrics(predictions, actual_returns, prefix="Return")
        else:
            return_metrics = {}
        
        # 整理结果
        results = {
            'MAE': price_metrics['Price MAE'],
            'RMSE': price_metrics['Price RMSE'],
            'MAPE': price_metrics['Price MAPE'] if price_metrics['Price MAPE'] is not None else 0,
            'Direction_Accuracy': return_metrics.get('Return Direction Accuracy', 
                                                   price_metrics.get('Price Direction Accuracy', 0)),
            'Correlation': price_metrics.get('Price Correlation', 0)
        }
        
        print(f"✅ 评估完成!")
        print(f"  价格MAE: ${results['MAE']:.2f}")
        print(f"  价格RMSE: ${results['RMSE']:.2f}")
        print(f"  价格MAPE: {results['MAPE']:.2f}%")
        print(f"  方向准确率: {results['Direction_Accuracy']:.2f}%")
        print(f"  相关系数: {results['Correlation']:.4f}")
        
        return results, price_metrics, return_metrics
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None



# ============================================
# Main
# ============================================
if __name__ == "__main__":

    import time
    start_time = time.time()  # Start timer
    print(f"🚀 增强版股票预测模型实验")
    print(f"🔧 设备: {device}")
    print(f"📋 实验参数:")
    for key, value in PARAMS.items():
        print(f"  {key}: {value}")
    
    # 运行实验
    results, price_metrics, return_metrics = Run_Experiment(PARAMS, "SP500")
    
    if results:
        print("💰 价格预测性能:")
        for name, value in price_metrics.items():
            if value is not None:
                if 'MAE' in name or 'RMSE' in name:
                    print(f"  {name}: ${value:.2f}")
                elif 'MAPE' in name or 'Accuracy' in name:
                    print(f"  {name}: {value:.2f}%")
                else:
                    print(f"  {name}: {value:.4f}")
        
        if PARAMS['USE_RETURNS'] and return_metrics:
            print(f"\n📊 收益率预测性能:")
            for name, value in return_metrics.items():
                if value is not None:
                    if 'Direction Accuracy' in name or 'MAPE' in name:
                        print(f"  {name}: {value:.2f}%")
                    else:
                        print(f"  {name}: {value:.4f}")
        
        print(f"\n🎯 核心指标总结:")
        print(f"  📈 方向准确率: {results['Direction_Accuracy']:.2f}%")
        print(f"  💲 价格MAE: ${results['MAE']:.2f}")
        print(f"  📏 价格RMSE: ${results['RMSE']:.2f}")
        print(f"  📊 价格MAPE: {results['MAPE']:.2f}%")
        print(f"  🔗 相关系数: {results['Correlation']:.4f}")
        
    else:
        print("❌ 实验失败，请检查数据文件和参数设置。")


    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


