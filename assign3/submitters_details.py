id_1 = "201014560"
id_2 = "057931354"
email = "raffilevy@gmail.com"

def get_details():
    if (not id_1) or (not id_2) or not (email):
        raise Exception("Missing submitters info")

    info = str.format("{}_{}      email: {}", id_1, id_2, email)

    return info