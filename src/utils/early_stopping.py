import sys, os
import copy


# class EarlyStopping:
#     def __init__(self, tolerance=5, min_delta=0):

#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, train_loss, validation_loss):
#         if (validation_loss - train_loss) > self.min_delta:
#             self.counter +=1
#             if self.counter >= self.tolerance:  
#                 self.early_stop = True

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0, restore_best_weights=True):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.best_model_optimizer = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, model, train_loss, validation_loss, optimizer):
        if self.best_loss == None:
            self.best_loss = validation_loss
            self.best_model = copy.deepcopy(model).to("cpu")
            self.best_model_optimizer = copy.deepcopy(optimizer)
        elif self.best_loss - validation_loss > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
            self.best_model_optimizer.load_state_dict(optimizer.state_dict())
        elif self.best_loss - validation_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.status = f'Stopped on iter {self.counter}'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                    optimizer.load_state_dict(self.best_model_optimizer.state_dict())
                return True
        # if (validation_loss - train_loss) > self.min_delta:
        #     self.counter +=1
        #     if self.counter >= self.tolerance:  
        #         return True
        self.status = f'{self.counter}/{self.tolerance}'
        return False




# early_stopping = EarlyStopping(tolerance=5, min_delta=10)

# for i in range(epochs):
    
#     print(f"Epoch {i+1}")
#     epoch_train_loss, pred = train_one_epoch(model, train_dataloader, loss_func, optimiser, device)
#     train_loss.append(epoch_train_loss)

#     # validation 
#     with torch.no_grad(): 
#        epoch_validate_loss = validate_one_epoch(model, validate_dataloader, loss_func, device)
#        validation_loss.append(epoch_validate_loss)
    
#     # early stopping
#     early_stopping(epoch_train_loss, epoch_validate_loss)
#     if early_stopping.early_stop:
#       print("We are at epoch:", i)
#       break