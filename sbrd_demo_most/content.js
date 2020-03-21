function appendSB(title, link) {
    // $(".sidebar.columns").append('<div class="column post-related post type-post status-publish format-standard has-post-thumbnail hentry"><div class="image_frame scale-with-grid"><div class="image_wrapper"><div class="mask"></div><img width="400" height="240" src="'+image_url+'" class="scale-with-grid wp-post-image" alt="" itemprop="image" srcset="'+image_url+' 400w, '+image_url+' 300w, '+image_url+' 243w, '+image_url+' 50w, '+image_url+' 125w" sizes="(max-width: 400px) 100vw, 400px"></a><div class="image_links double"><a href="'+image_url+'" class="zoom" rel="prettyphoto"><i class="icon-search"></i></a><a href="'+link+'" class="link"><i class="icon-link"></i></a></div></div></div><div class="desc"><h4><a href="'+link+'">'+title+'</a></h4><hr class="hr_color"></div></div>');      // Append the new elements
    // $(".sidebar.columns").append('<br>')
    $('#lyr1').append('<li><span class="SpanStar"></span><a href="https://www.most.gov.vn' + link + '">' + title +  '<font>&nbsp;&nbsp;(sb recommend)</font></a><span class="Border_Line"></span></li>')
}

function appendStat(title, link, class_name) {
  $('.'+class_name).append('<li> <a href="https://www.most.gov.vn' + link + '">' + title +  ' <span>(stat recommended)</span></a></li>')
}

var array = JSON.parse(window.localStorage.cly_view_history);
var request = "";
array.forEach(function(entry) {
    request += entry['name'] + " ";
});
request += window.location.pathname;

$( document ).ready(function(event){
    $.ajax({
      type : "POST",
      // url : "http://203.162.10.123:5006/api/dter_inference",
      url : "https://21a86e38.ap.ngrok.io/api/dter_inference",
      data : JSON.stringify({"list_url":request}),
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
      success: function(respone){
          // $(".sections_group").css('float', 'left');
          // $(".sections_group").css('width', '70%');
          // $(".sidebar.columns").css('width', '30%');
          // $(".sidebar.columns").css('background-color', 'white');
          // $(".sidebar.columns").css('border', '2px solid red');
          // $(".sidebar.columns").html("<h4 style='padding-top:30px; text-align:center; color:red;'>RECOMMENDED ITEMS</h4>");
          var i = 0;
          var o = ['Hot-Trend', 'Most-Popular', 'Session-based-Recommend'];
          $('.group_header_link').each(function() {
            $(this).html(o[i]);
            i += 1;
          });
          var i = 0;
          $('.othernews_fullwidth').each(function() {
            $(this).addClass(o[i]);
            $(this).html('');
            i += 1;
          });
          $('#lyr1').html('');
          respone["data"]["sb"].forEach(function(entry) {
            // appendText("http://portal.ptit.edu.vn/wp-content/uploads/2019/01/1-10.jpg", entry[1], entry[0]);
            appendSB(entry[1], entry[0]);
          });
          respone["data"]["stat"].forEach(function(entry) {
            appendStat(entry[1], entry[0], entry[2]);
          });
        },
      error: function(XMLHttpRequest, textStatus, errorThrown) {
        alert("some error " + String(errorThrown) + String(textStatus) + String(XMLHttpRequest.responseText));
        }
      });
    });


