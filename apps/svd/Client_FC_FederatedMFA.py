from apps.svd.FC_Federated_MFA import FCFederatedMFA

class ClientFCFederatedMFA(FCFederatedMFA):
    def __init__(self):
        self.dummy = None
        self.coordinator = False
        FCFederatedMFA.__init__(self)