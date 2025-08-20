// 初始化日志函数
function logToConsole(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    console.log(logMessage);
    
    // 可选: 发送日志到后端API
    // fetch('/api/log', {
    //     method: 'POST',
    //     headers: {'Content-Type': 'application/json'},
    //     body: JSON.stringify({message, level, timestamp})
    // });
}

// 初始化评估界面
document.addEventListener('DOMContentLoaded', function() {
    logToConsole('Initializing judge interface');
    // 全局变量
    let currentImageIndex = 0;
    let imagesToJudge = [];
    let selectedWinner = null;
    
    // 图像选择处理
    function handleImageSelection(winner) {
        imageA.classList.remove('selected');
        imageB.classList.remove('selected');
        selectedWinner = winner;
        document.getElementById(`image-${winner.toLowerCase()}`).classList.add('selected');
        logToConsole(`Selected image ${winner} as winner`);
    }
    
    // DOM元素
    const imageA = document.getElementById('image-a');
    const imageB = document.getElementById('image-b');
    const promptTextElement = document.getElementById('prompt-text');
    const modelNameElement = document.getElementById('model-name');
    const seedValueElement = document.getElementById('seed-value');
    const skipButton = document.getElementById('skip-btn');
    const submitButton = document.getElementById('submit-btn');
    const progressCurrent = document.getElementById('progress-current');
    const progressTotal = document.getElementById('progress-total');
    
    // 初始化评估数据
    async function initJudge() {
        try {
            // 从API获取下一组待比较图像
            logToConsole('Fetching next image pair from API');
            const response = await fetch('/api/prompts/${prompt_id}/next_comparison');
            const data = await response.json();
            imagesToJudge = [{
                prompt_id: data.prompt_id,
                prompt_text: data.prompt_text,
                seed_a: data.choice_a.seed,
                image_a_path: data.choice_a.image_url,
                seed_b: data.choice_b.seed,
                image_b_path: data.choice_b.image_url
            }];
            logToConsole(`Received ${imagesToJudge.length} images to judge`);
            
            if (imagesToJudge.length > 0) {
                loadCurrentImage();
            } else {
                logToConsole('No images available for judging');
                showNoImagesMessage();
            }
        } catch (error) {
            logToConsole(`Error loading judge queue: ${error.message}`, 'error');
            alert('加载评估队列失败，请稍后重试');
        }
    }
    
    // 加载当前图像对
    function loadCurrentImage() {
        const currentImage = imagesToJudge[currentImageIndex];
        imageA.src = currentImage.image_a_path;
        imageB.src = currentImage.image_b_path;
        promptTextElement.textContent = currentImage.prompt_text;
        modelNameElement.textContent = currentImage.model_name;
        seedValueElement.textContent = currentImage.seed_a + ' vs ' + currentImage.seed_b;
        
        // 更新进度显示
        progressCurrent.textContent = currentImageIndex + 1;
        progressTotal.textContent = imagesToJudge.length;
        
        // 重置选择状态
        selectedWinner = null;
        imageA.classList.remove('selected');
        imageB.classList.remove('selected');
    }
    
    
    // 跳过按钮点击事件
    skipButton.addEventListener('click', function() {
        if (currentImageIndex < imagesToJudge.length - 1) {
            currentImageIndex++;
            loadCurrentImage();
        } else {
            showNoImagesMessage();
        }
    });
    
    // 提交按钮点击事件
    submitButton.addEventListener('click', async function() {
        if (!selectedWinner) {
            alert('请先选择更好的图像');
            return;
        }
        
        try {
            const currentImage = imagesToJudge[currentImageIndex];
            logToConsole(`Submitting judgment for pair ${currentImage.id}`);
            const response = await fetch('/api/judgments', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt_id: currentImage.prompt_id,
                    winner_seed: selectedWinner === 'A' ? currentImage.seed_a : currentImage.seed_b,
                    loser_seed: selectedWinner === 'A' ? currentImage.seed_b : currentImage.seed_a,
                    user_id: "current_user" // TODO: Replace with actual user ID
                })
            });
            
            const result = await response.json();
            if (result.success) {
                // 移动到下一张图像
                if (currentImageIndex < imagesToJudge.length - 1) {
                    currentImageIndex++;
                    loadCurrentImage();
                } else {
                    showNoImagesMessage();
                }
            } else {
                alert('提交评估结果失败: ' + result.message);
            }
        } catch (error) {
            logToConsole(`Error submitting rating: ${error.message}`, 'error');
            alert('提交评分失败，请稍后重试');
        }
    });
    
    // 显示没有更多图像的消息
    function showNoImagesMessage() {
        document.querySelector('.image-pair').style.display = 'none';
        document.querySelector('.evaluation-controls').style.display = 'none';
        const message = document.createElement('p');
        message.textContent = '评估完成！感谢您的参与';
        document.querySelector('.prompt-info').appendChild(message);
    }
    
    // 图像点击事件
    imageA.addEventListener('click', () => handleImageSelection('A'));
    imageB.addEventListener('click', () => handleImageSelection('B'));
    
    // 键盘快捷键
    document.addEventListener('keydown', (e) => {
        if (e.key === '1') handleImageSelection('A');
        if (e.key === '2') handleImageSelection('B'); 
        if (e.key === ' ') submitButton.click(); // Space to submit
        if (e.key === 's') skipButton.click();  // 's' to skip
    });
    
    // 初始化评估界面
    initJudge();
});
