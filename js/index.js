fetch("http://127.0.0.1:3001/api/classify/text", {
  method: "POST",
  headers: new Headers({ "content-type": "application/json" }),
  body: JSON.stringify({
    text: "i feel like my irritable sensitive combination skin has finally met it s match.",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
