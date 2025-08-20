// 系统设置界面逻辑
document.addEventListener('DOMContentLoaded', function() {
    // DOM元素
    const promptFile = document.getElementById('prompt-file');
    const initModelSelect = document.getElementById('init-model-select');
    const initBtn = document.getElementById('init-btn');
    const initStatus = document.getElementById('init-status');
    const modelSelect = document.getElementById('model-select');
    const stepsInput = document.getElementById('steps-input');
    const guidanceInput = document.getElementById('guidance-input');
    const loraList = document.getElementById('lora-list');
    const loraUpload = document.getElementById('lora-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const saveBtn = document.getElementById('save-settings');
    
    // 初始化设置界面
    async function initSettings() {
        try {
            // 加载当前设置
            const response = await fetch('/api/settings');
            const settings = await response.json();
            
            // 更新UI
            modelSelect.value = settings.model_name || 'runwayml/stable-diffusion-v1-5';
            stepsInput.value = settings.num_inference_steps || 20;
            guidanceInput.value = settings.guidance_scale || 7.5;
            
            // 加载LoRA列表
            await loadLoraList();
        } catch (error) {
            console.error('加载设置失败:', error);
            alert('加载系统设置失败，请稍后重试');
        }
    }
    
    // 加载LoRA列表
    async function loadLoraList() {
        try {
            const response = await fetch('/api/lora/list');
            const loras = await response.json();
            
            loraList.innerHTML = '';
            if (loras.length === 0) {
                loraList.innerHTML = '<p>没有可用的LoRA模型</p>';
                return;
            }
            
            const ul = document.createElement('ul');
            loras.forEach(lora => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <div class="lora-item">
                        <span>${lora.name}</span>
                        <span>缩放: ${lora.scale}</span>
                        <button class="delete-btn" data-id="${lora.id}">删除</button>
                    </div>
                `;
                ul.appendChild(li);
            });
            
            loraList.appendChild(ul);
            
            // 添加删除按钮事件
            document.querySelectorAll('.delete-btn').forEach(btn => {
                btn.addEventListener('click', async function() {
                    if (confirm('确定要删除这个LoRA模型吗？')) {
                        try {
                            const response = await fetch(`/api/lora/delete/${this.dataset.id}`, {
                                method: 'DELETE'
                            });
                            const result = await response.json();
                            if (result.success) {
                                await loadLoraList();
                            } else {
                                alert('删除失败: ' + result.message);
                            }
                        } catch (error) {
                            console.error('删除LoRA失败:', error);
                            alert('删除LoRA失败，请稍后重试');
                        }
                    }
                });
            });
        } catch (error) {
            console.error('加载LoRA列表失败:', error);
            alert('加载LoRA列表失败，请稍后重试');
        }
    }
    
    // LoRA上传处理
    uploadBtn.addEventListener('click', async function() {
        const file = loraUpload.files[0];
        if (!file) {
            alert('请先选择LoRA模型文件');
            return;
        }
        
        const formData = new FormData();
        formData.append('lora_file', file);
        
        try {
            const response = await fetch('/api/lora/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (result.success) {
                alert('LoRA上传成功');
                await loadLoraList();
                loraUpload.value = '';
            } else {
                alert('上传失败: ' + result.message);
            }
        } catch (error) {
            console.error('上传LoRA失败:', error);
            alert('上传LoRA失败，请稍后重试');
        }
    });
    
    // 保存设置
    saveBtn.addEventListener('click', async function() {
        const settings = {
            model_name: modelSelect.value,
            num_inference_steps: parseInt(stepsInput.value),
            guidance_scale: parseFloat(guidanceInput.value)
        };
        
        try {
            const response = await fetch('/api/settings/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            const result = await response.json();
            if (result.success) {
                alert('设置保存成功');
            } else {
                alert('保存失败: ' + result.message);
            }
        } catch (error) {
            console.error('保存设置失败:', error);
            alert('保存设置失败，请稍后重试');
        }
    });
    
    // 初始化系统
    initBtn.addEventListener('click', async function() {
        const file = promptFile.files[0];
        if (!file) {
            updateStatus('请选择提示词文件');
            return;
        }

        updateStatus('正在初始化系统...');
        
        try {
            const formData = new FormData();
            formData.append('prompt_file', file);
            formData.append('model_name', initModelSelect.value);

            const response = await fetch('/api/initialize', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.success) {
                updateStatus('初始化成功！正在加载评估队列...');
                
                // 获取评估队列
                const queueResponse = await fetch('/api/judge/queue');
                const queueData = await queueResponse.json();
                
                if (queueData.items.length > 0) {
                    updateStatus('初始化成功！正在跳转到评估页面...');
                    setTimeout(() => {
                        window.location.href = 'judge.html';
                    }, 2000);
                } else {
                    updateStatus('错误: 评估队列为空');
                }
            } else {
                updateStatus('初始化失败: ' + result.message);
            }
        } catch (error) {
            console.error('初始化失败:', error);
            updateStatus('初始化失败: ' + error.message);
        }
    });

    function updateStatus(message) {
        initStatus.innerHTML = `<p>${new Date().toLocaleTimeString()}: ${message}</p>`;
    }

    // 初始化设置界面
    initSettings();
});
