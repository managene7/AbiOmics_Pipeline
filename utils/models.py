import torch
import torch.nn as nn

# MLP model_________________________________
# class mlp_model(nn.Module): #  mlp layers
#     def __init__(self, dim1, label_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#         nn.Linear(dim1, dim1*1),
#         nn.ReLU(),
#         # nn.Linear(dim1*1, dim1*1),
#         # nn.ReLU(),
#         nn.Linear(dim1*1, dim1*1),
#         nn.ReLU(),
#         nn.Linear(dim1*1, label_dim),
#         )

#     def forward(self,x):
#         x = self.mlp(x)
#         return x

# class mlp_model(nn.Module): #  mlp layers
#     def __init__(self, dim1=0, label_dim=5):
#         super().__init__()
#         self.input=nn.Sequential(
#             nn.Linear(dim1, dim1*1),
#             nn.ReLU(),
#         )
#         self.hidden=nn.Sequential(
#             #<------- modify model from here ------->        

#             nn.Linear(dim1*1, dim1*1),
#             nn.ReLU(),
           
#         )
#         self.classifier=nn.Sequential(
#             nn.Linear(dim1*1, label_dim),
#         )
        
#     def forward(self,x):
#         first_layer_feature=self.input(x)
#         hidden_feature=self.hidden(first_layer_feature)
        
#         merged_feature=first_layer_feature+hidden_feature
#         last_feature=self.hidden(merged_feature)
#         logits=self.classifier(last_feature)

#         logits=torch.sigmoid(logits)######################################################

#         return logits

class mlp_model(nn.Module): #  mlp layers
    def __init__(self, dim1=0, label_dim=5):
        super().__init__()
        # self.input=nn.Sequential(
        #     nn.Linear(dim1, dim1*1),
        #     nn.ReLU(),
        # )
        # self.hidden=nn.Sequential(
        #     #<------- modify model from here ------->        

        #     nn.Linear(dim1*1, dim1*1),
        #     nn.ReLU(),
           
        # )
        self.classifier=nn.Sequential(
            nn.Linear(dim1 * 1, label_dim),
            # nn.ReLU(),
        )
        
    def forward(self,x):
        # first_layer_feature=self.input(x)
        # hidden_feature=self.hidden(first_layer_feature)
        
        # merged_feature=first_layer_feature+hidden_feature
        # last_feature=self.hidden(hidden_feature)
        logits = self.classifier(x)

        logits = torch.sigmoid(logits)######################################################

        return logits
