// 平滑滚动
document.addEventListener('DOMContentLoaded', function() {
    // 处理导航链接的平滑滚动
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80; // 考虑固定导航栏的高度
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // 高亮当前部分的导航项
    const sections = document.querySelectorAll('.section[id]');
    const navItems = document.querySelectorAll('.nav-link[href^="#"]');

    function highlightNavItem() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (window.pageYOffset >= sectionTop && 
                window.pageYOffset < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${current}`) {
                item.classList.add('active');
            }
        });
    }

    // 监听滚动事件
    window.addEventListener('scroll', highlightNavItem);
    
    // 初始化时也执行一次
    highlightNavItem();
});

// 添加页面加载动画
window.addEventListener('load', function() {
    const elements = document.querySelectorAll('.feature-card, .module-card, .step');
    
    elements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        
        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// 导航栏透明度处理
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
    }
});

// 工具函数：复制代码功能
function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(codeBlock => {
        const pre = codeBlock.parentElement;
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = '复制';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: #667eea;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        `;
        
        pre.style.position = 'relative';
        pre.appendChild(button);
        
        button.addEventListener('click', () => {
            navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                button.textContent = '已复制!';
                setTimeout(() => {
                    button.textContent = '复制';
                }, 2000);
            });
        });
    });
}

// 页面加载完成后添加复制按钮
document.addEventListener('DOMContentLoaded', addCopyButtons);
