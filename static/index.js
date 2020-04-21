let btn = document.getElementById('submit')

btn.addEventListener('click',generateText);

const get_text = async (initial_text,length) => {
    out = document.getElementById('output_content')
    var url = new URL("http://127.0.0.1:5000/generate_text/"),
        params = {"initial_text":initial_text, "length":length}
    Object.keys(params).forEach(key => url.searchParams.append(key, params[key]))
    const response = await fetch(url, {
      method: 'POST',
      mode: 'cors',
      headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
            },
    });
    const myJson = await response.json(); //extract JSON from the http response
    out.innerHTML = myJson.output
  }

function generateText(){
    document.getElementById('output').style.display = 'block'
    initial_text = document.getElementById('input').value
    length = document.getElementById('length').value
    get_text(initial_text,length)
}