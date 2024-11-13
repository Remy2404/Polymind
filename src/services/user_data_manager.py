class UserDataManager:
    def __init__(self):
        self.user_contexts = {}
    
    def get_user_context(self, user_id: int) -> list:
        return self.user_contexts.get(user_id, [])
    
    def update_user_context(self, user_id: int, message: str, response: str):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        
        self.user_contexts[user_id].append({"role": "user", "content": message})
        self.user_contexts[user_id].append({"role": "assistant", "content": response})
        
        # Keep only last 10 messages for contextclass UserDataManager:
    def __init__(self):
        self.user_contexts = {}
        self.user_settings = {}
    
    def initialize_user(self, user_id: int):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {
                'markdown_enabled': True,
                'code_suggestions': True
            }
    
    def get_user_context(self, user_id: int) -> list:
        return self.user_contexts.get(user_id, [])
    
    def update_user_context(self, user_id: int, message: str, response: str):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        
        self.user_contexts[user_id].append({"role": "user", "content": message})
        self.user_contexts[user_id].append({"role": "assistant", "content": response})
        
        # Keep only last 10 messages for context
        if len(self.user_contexts[user_id]) > 20:
            self.user_contexts[user_id] = self.user_contexts[user_id][-20:]
    
    def reset_user_data(self, user_id: int):
        self.user_contexts[user_id] = []
    
    def get_user_settings(self, user_id: int) -> dict:
        return self.user_settings.get(user_id, {
            'markdown_enabled': True,
            'code_suggestions': True
        })

        if len(self.user_contexts[user_id]) > 20:
            self.user_contexts[user_id] = self.user_contexts[user_id][-20:]
