import torch
import numpy as np
import pandas as pd
from .models.registry import MODEL_REGISTRY
from .callbacks import EarlyStopping
from .meta.meta_learner import WeeklyMetaLearner
from .utils.metrics import calculate_metrics

class StackingOrchestrator:
    def __init__(self, base_model_names, node_features, num_nodes, device='cuda', **model_kwargs):
        self.base_model_names = base_model_names
        self.device = device
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.model_kwargs = model_kwargs

        # Validate models
        for name in base_model_names:
            if name not in MODEL_REGISTRY:
                raise ValueError(f"Model {name} not registered.")

    def _init_base_model(self, name):
        ModelClass = MODEL_REGISTRY[name]
        # AGCRN requires num_nodes arg
        if name == 'agcrn':
             model = ModelClass(self.node_features, num_nodes=self.num_nodes, **self.model_kwargs)
        else:
             model = ModelClass(self.node_features, **self.model_kwargs)
        return model.to(self.device)

    def _train_base_model(self, model, model_name, data_train, max_epochs, lr, patience):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping(patience=patience)
        
        # AGCRN embedding init
        e = None
        if model_name == 'agcrn':
            e = torch.empty(self.num_nodes, 1).to(self.device)
            torch.nn.init.xavier_uniform_(e)

        model.train()
        for epoch in range(max_epochs):
            cost = 0
            h, c = None, None
            
            for snapshot in data_train:
                x = snapshot.x.to(self.device)
                y = snapshot.y.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                
                # Handle edge attributes if present
                edge_weight = getattr(snapshot, 'edge_attr', None)
                if edge_weight is not None: 
                    edge_weight = edge_weight.to(self.device)

                # Forward pass routing
                if model_name == 'agcrn':
                    y_hat, h = model(x.view(1, self.num_nodes, -1), e, h)
                    if isinstance(h, torch.Tensor): h = h.detach()
                    loss = torch.mean((y_hat - y)**2)
                elif model_name in ['lrgcn', 'gclstm', 'dygrencoder', 'gconvlstm']:
                    y_hat, h, c = model(x, edge_index, edge_weight, h, c)
                    if isinstance(h, torch.Tensor): h = h.detach()
                    if isinstance(c, torch.Tensor): c = c.detach()
                    loss = torch.mean(torch.abs(y_hat - y))
                else:
                    y_hat = model(x, edge_index, edge_weight)
                    loss = torch.mean(torch.abs(y_hat - y))
                
                cost += loss

            cost /= len(data_train)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            early_stopper(cost.item())
            if early_stopper.early_stop:
                break
        return model

    def _predict_base_model(self, model, model_name, data_loader):
        model.eval()
        preds = []
        actuals = []
        
        h, c = None, None
        e = None
        if model_name == 'agcrn':
            e = torch.empty(self.num_nodes, 1).to(self.device)
            torch.nn.init.xavier_uniform_(e)

        with torch.no_grad():
            for snapshot in data_loader:
                x = snapshot.x.to(self.device)
                y = snapshot.y.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = getattr(snapshot, 'edge_attr', None)
                if edge_weight is not None: 
                    edge_weight = edge_weight.to(self.device)
                
                if model_name == 'agcrn':
                    y_hat, h = model(x.view(1, self.num_nodes, -1), e, h)
                elif model_name in ['lrgcn', 'gclstm', 'dygrencoder', 'gconvlstm']:
                    y_hat, h, c = model(x, edge_index, edge_weight, h, c)
                else:
                    y_hat = model(x, edge_index, edge_weight)
                
                preds.extend(y_hat.cpu().numpy().flatten())
                actuals.extend(y.cpu().numpy().flatten())
                
        return np.array(preds), np.array(actuals)

    def run_nested_cross_validation(self, data_list, 
                                    initial_train_window=28, 
                                    meta_train_window=14,    
                                    meta_test_window=14,     
                                    stride=7,                
                                    base_epochs=100, base_lr=0.001,
                                    meta_lag=3, meta_trials=20):
        
        total_len = len(data_list)
        current_t = initial_train_window
        
        all_meta_predictions = []
        all_meta_actuals = []
        fold = 1
        
        while current_t + meta_train_window + meta_test_window <= total_len:
            print(f"\n>>> Processing Split {fold} (Time {current_t})")
            
            # Define periods
            idx_base_train = range(0, current_t)
            idx_meta_train = range(current_t, current_t + meta_train_window)
            idx_meta_test  = range(current_t + meta_train_window, current_t + meta_train_window + meta_test_window)
            
            data_base_train = [data_list[i] for i in idx_base_train]
            data_meta_train = [data_list[i] for i in idx_meta_train]
            data_meta_test  = [data_list[i] for i in idx_meta_test]
            
            fold_meta_train_X = {}
            fold_meta_test_X = {}
            fold_actuals_train = None
            fold_actuals_test = None
            
            # Train Base Learners
            for m_name in self.base_model_names:
                print(f"  [Base] Training {m_name}...")
                model = self._init_base_model(m_name)
                model = self._train_base_model(model, m_name, data_base_train, base_epochs, base_lr, patience=10)
                
                # Predict Period B (Meta Train)
                preds_B, acts_B = self._predict_base_model(model, m_name, data_meta_train)
                fold_meta_train_X[m_name] = preds_B
                fold_actuals_train = acts_B
                
                # Predict Period C (Meta Test)
                preds_C, acts_C = self._predict_base_model(model, m_name, data_meta_test)
                fold_meta_test_X[m_name] = preds_C
                fold_actuals_test = acts_C
            
            # Train Meta Learner
            print(f"  [Meta] Tuning & Training Stacking Model...")
            df_train_X = pd.DataFrame(fold_meta_train_X)
            s_train_y = pd.Series(fold_actuals_train, name='actual')
            
            df_test_X = pd.DataFrame(fold_meta_test_X)
            s_test_y = pd.Series(fold_actuals_test, name='actual')
            
            meta_learner = WeeklyMetaLearner(lag_window=meta_lag, n_trials=meta_trials)
            meta_learner.tune_and_train(df_train_X, s_train_y)
            
            # Final Predict
            final_preds, final_y = meta_learner.predict(df_test_X, s_test_y)
            
            all_meta_predictions.extend(final_preds)
            all_meta_actuals.extend(final_y)
            
            metrics = calculate_metrics(final_preds, final_y)
            print(f"  [Result] Fold {fold} RMSE: {metrics['RMSE']:.4f}")
            
            current_t += stride
            fold += 1
            
        return np.array(all_meta_predictions), np.array(all_meta_actuals)