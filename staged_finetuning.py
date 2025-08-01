class StagedFineTuner:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        
    def stage1_train(self, data_loader, epochs=10, warmup=5):
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze RPN and ROI heads
        for module in [self.model.rpn, self.model.roi_heads]:
            for param in module.parameters():
                param.requires_grad = True
                
        self._setup_optimizer(lr=1e-4, warmup=warmup)
        self._train_loop(data_loader, epochs)
    
    def stage2_train(self, data_loader, epochs=10):
        # Unfreeze deeper layers
        for layer in [self.model.backbone.body.layer3, 
                     self.model.backbone.body.layer4]:
            for param in layer.parameters():
                param.requires_grad = True
                
        self._setup_optimizer(lr=1e-5)
        self._train_loop(data_loader, epochs)
    
    def _setup_optimizer(self, lr, warmup=None):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
        
        if warmup:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda e: e/warmup if e < warmup else 1)
        else:
            self.scheduler = None
    
    def _train_loop(self, data_loader, epochs):
        for epoch in range(epochs):
            train_one_epoch(self.model, self.optimizer, data_loader, ...)
            if self.scheduler:
                self.scheduler.step()