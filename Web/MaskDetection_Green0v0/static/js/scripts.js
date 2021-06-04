// 여기가 경로!
const video = document.getElementById('video')  

Promise.all([]).then(startVideo) //함수를 실행시키는 코드!

function startVideo() {
    navigator.getUserMedia(
        { video: {}},
        stream => video.srcObject = stream,
        err => console.error(err)
    )
}
