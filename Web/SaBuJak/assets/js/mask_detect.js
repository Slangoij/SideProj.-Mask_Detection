function startVideo() {
    const video = document.querySelector('video');
    const img = document.querySelector('img');

    // play webcam
    navigator.mediaDevices.getUserMedia({video: {width: 500, height: 281}})
    .then((stream) => video.srcObject = stream);

    const getFrame = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const data = canvas.toDataURL('image/png');
        return data;
    }

    const WS_URL = 'ws://localhost:5500/ws';   // <<<<<<< 이 부분 port번호 맞추기
    const FPS = 3;
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log(`Connected to ${WS_URL}`);
        setInterval(() => {
            ws.send(getFrame());
        }, 1000 / FPS);
    }

    ws.onmessage = message => {
        const msg = message.data;
        
        img.src = msg.substring(0, msg.length -3);
        const dangerRate = msg.substring(msg.length -3);

        const bar = document.getElementById('myBar');
        bar.style.width = dangerRate + '%';
        document.getElementById('label').innerHTML = dangerRate * 1 + '%'
    }
}