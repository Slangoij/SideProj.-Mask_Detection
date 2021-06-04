
//html2canvas 이용해 웹화면 스샷 base64포맷으로 변환 + ajax로 data보내기
$(function(){
    $("#save").click(function() {
        html2canvas($('#retrocon').get(0)).then(function(canvas){
            var data = canvas.toDataURL();
            console.log(data);

            //ajax통신
            $.ajax({
                type: 'POST',
                url: 'image',
                data: {data:data},
                success: function(result) {
                    var filename = result['filename'];
                    console.log(filename);
                },
                error: function(e) { alert("에러발생");}
            });

        });
    });
});