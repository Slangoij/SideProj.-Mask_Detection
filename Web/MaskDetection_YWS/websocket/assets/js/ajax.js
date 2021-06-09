$.ajax({
    type: 'POST',
    url: '/detect/temp/',
    data: {data:data},
    success: function(result) {
        var filename = result['filename'];
        console.log("통신성공");
    },
    error: function(e) { console.log("에러발생");}
});

// //ajax로 data보내기
// $(function(){
//     $("#save").click(function() {
//         html2canvas($('#screenshot').get(0)).then(function(canvas){
//             var data = canvas.toDataURL();
//             console.log(data);

//             //ajax통신
//             $.ajax({
//                 type: 'POST',
//                 url: '/detect/temp/',
//                 data: {data:data},
//                 success: function(result) {
//                     var filename = result['filename'];
//                     console.log(filename);
//                     alert("통신성공");
//                 },
//                 error: function(e) { alert("에러발생");}
//             });

//         });
//     });
// });