var colorvalue = "#f3ece0";
var colorvalue_sole = "#f3ece0";

function grabData() {
  return fetch("/static/pantoneColor.json")
    .then(response => response.json());
}

grabData().then(data => {
  $.each(data, function (key, value) {
    $('#shoePantone')
      .append($("<option></option>")
        .attr("value", value)
        .attr("class", key)
        .text(key));
  });
  
  $.each(data, function (key, value) {
    $('#solePantone')
      .append($("<option></option>")
        .attr("value", value)
        .attr("class", key)
        .text(key));
  });
});

$('#shoePantone').change(function () {
  var optionSelected = $(this).find("option:selected");
  colorvalue = optionSelected.val();
  $(this).css('background-color', colorvalue);
});

$('#solePantone').change(function () {
  var optionSelected = $(this).find("option:selected");
  colorvalue_sole = optionSelected.val();
  $(this).css('background-color', colorvalue_sole);
});

var base64string = "";
var base64recvstring = "";

$("#register_imgfile").change(function () {
  readURL(this);
});

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    reader.onload = function (e) {}
    reader.readAsDataURL(input.files[0]); // convert to base64 string
  }
}

if (window.File && window.FileReader && window.FileList && window.Blob) {
  document.getElementById('register_imgfile').addEventListener('change', handleFileSelect, false);
} else {
  alert('The File APIs are not fully supported in this browser.');
}

function handleFileSelect(evt) {
  var f = evt.target.files[0]; // FileList object
  var reader = new FileReader();
  // Closure to capture the file information.
  reader.onload = (function (theFile) {
    return function (e) {
      var binaryData = e.target.result;
      base64string = btoa(binaryData);
      postuploadImageFile();
    };
  })(f);
  // Read in the image file as a data URL.
  reader.readAsBinaryString(f);
}


function postImageFile() {
  var base64 = base64string;
  base64recvstring = "";
  // var username = $("#register_username").val();
  $.post("RmbkPortrait",
    {
      portraitphoto: base64,
      color: colorvalue,
      color_sole: colorvalue_sole
    },
    function (data, status) {
      if (data == "") {
        alert("Failed processing !");
      } else {
        $('#register_photo2').attr('src', data);
        base64recvstring = data;
      }
    });
};

function onSetColorClick() {
  postColor();
}

function postColor() {
  var base64 = base64string;
  base64recvstring = "";
  $.post("COLOR_Post",
    {
      color: colorvalue,
      color_sole: colorvalue_sole
    },
    function (data, status) {
      if (data == "") {
        alert("Failed processing !");
      } else {
        $('#register_photo2').attr('src', data);
        base64recvstring = data;
      }
    });
};

function postuploadImageFile() {
  var base64 = base64string;
  base64recvstring = "";
  $.post("imgPortrait",
    {
      portraitphoto: base64
    },
    function (data, status) {
      if (data == "") {
        alert("Failed processing !");
      } else {
        $('#register_photo').attr('src', data);
        base64recvstring = data;
      }
    });
};

function onUploadClick() {
  $('#register_imgfile').trigger('click');
}

function onProcessClick() {
  postImageFile();
}


function setCookie(name, value, days) {
  var expires = "";
  if (days) {
    var date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    expires = "; expires=" + date.toUTCString();
  }
  document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

function getCookie(name) {
  var nameEQ = name + "=";
  var ca = document.cookie.split(';');
  for (var i = 0; i < ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0) == ' ') c = c.substring(1, c.length);
    if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
}

function eraseCookie(name) {
  document.cookie = name + '=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
}