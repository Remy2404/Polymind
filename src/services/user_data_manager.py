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
        
        # Keep only last 10 messages for context
        if len(self.user_contexts[user_id]) > 20:
            self.user_contexts[user_id] = self.user_contexts[user_id][-20:]
