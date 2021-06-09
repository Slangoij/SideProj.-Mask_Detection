const video = document.querySelector('video');
const img = document.querySelector('img');
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
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
    // set the base64 string to the src tag of the image
    var mess = message.data

    // 이미지 소스에 분석한 이미지 삽입
    img.src = mess.substring(0, mess.length - 3);
    // 프로그래스바=>myBar
    var elem = document.getElementById("myBar");

    elem.style.width = mess.substring(mess.length - 3) + '%';

    dangerRate = mess.substring(mess.length - 3)
    // if (dangerRate > 80) {
    //     elem.style.background = red
    // }elseif (dangerRage > 50) {

    // }
    // else {

    // }
    document.getElementById("label").innerHTML = dangerRate * 1 + '%';
    console.log(img.src)
}