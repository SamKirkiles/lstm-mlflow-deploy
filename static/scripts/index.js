function generate(book) {
    chars = $("#characterCount").val()
    if (!/[0-9]+/.test(chars) || parseInt(chars) > 1000 || parseInt(chars) < 1) {
        alert("Number of characters must be from 1 to 1000!")
        $("#characterCount").focus()
        return
    }
    seed = $("#seedValue").val()
    if (!/[a-z]/.test(seed) || seed.length != 1) {
        alert("Seed value must be a-z!")
        $("#seedValue").focus()
        return
    }
    $.ajax({
        url: "/predict",
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        data: JSON.stringify({
            "output_length": [chars],
            "book": [book],
            "seed": [seed]
        }),
        success: function(result) {
            $("#textHallucinationArea").val(result);
        },
        error: function(xhr, status, error) {
            alert("Error! Check console logs.")
            console.log(error)
        }
    });
}