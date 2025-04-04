class MCPService:
    def __init__(self):
        self.context = {}
    def getStatus(self):
        return {
            "status" : "running",
            "current_context" : self.context
        }
    def setContext(self, contextData : dict) :
        self.context.update(contextData)