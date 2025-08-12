"""
Example JavaScript client code for connecting to the Moodle AI Assistant backend.

This would be integrated into the Moodle block plugin.
"""

javascript_client_example = '''
// Moodle AI Assistant Client Example
class MoodleAIAssistantClient {
    constructor(baseUrl = 'http://127.0.0.1:8000') {
        this.baseUrl = baseUrl;
    }

    // Check server health
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }

    // Get system status (RAG vs Generation mode)
    async getSystemStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/status`);
            return await response.json();
        } catch (error) {
            console.error('Status check failed:', error);
            return null;
        }
    }

    // Send chat message with streaming response
    async sendMessage(message, history = [], onChunk = null) {
        try {
            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: history
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { value, done } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        
                        if (data === '[DONE]') {
                            return fullResponse;
                        } else if (data.startsWith('ERROR:')) {
                            throw new Error(data);
                        } else {
                            fullResponse = data; // Each chunk contains the full response so far
                            if (onChunk) onChunk(data);
                        }
                    }
                }
            }
            
            return fullResponse;

        } catch (error) {
            console.error('Chat request failed:', error);
            throw error;
        }
    }
}

// Usage example in Moodle block
async function initializeMoodleAssistant() {
    const client = new MoodleAIAssistantClient();
    
    // Check if server is healthy
    const health = await client.checkHealth();
    if (!health) {
        console.error('Backend server not available');
        return;
    }
    
    // Get system mode
    const status = await client.getSystemStatus();
    console.log(`AI Assistant Mode: ${status.mode}`);
    console.log(`Documents available: ${status.documents_folder_exists}`);
    
    // Send a message with streaming response
    const userMessage = "Hello, can you help me with this course?";
    const history = []; // Previous conversation history
    
    let currentResponse = '';
    
    try {
        const response = await client.sendMessage(
            userMessage, 
            history,
            (chunk) => {
                // This function is called for each streaming chunk
                currentResponse = chunk;
                document.getElementById('ai-response').innerHTML = currentResponse;
            }
        );
        
        console.log('Final response:', response);
        
        // Update history for next message
        history.push(
            { role: 'user', content: userMessage },
            { role: 'assistant', content: response }
        );
        
    } catch (error) {
        console.error('Error getting AI response:', error);
    }
}
'''

print("JavaScript client example:")
print(javascript_client_example)
