"""Full-Craftax texture tables, baked from craftax 1.6.1 at tile sizes (10, 16) so the renderer imports no jax.

uint8 and lossless: craftax's ``load_texture`` forces every alpha to exactly 0 or 1 and resizes with
``Image.NEAREST``. The only non-integral texture is ``night_noise_intensity_texture``, a function of
the frame size built at runtime by :func:`stable_envs.envs.craftax.full.night_vignette`.

Generated data; do not hand-edit. A new craftax version needs new tables and a new parity measurement.
"""

from __future__ import annotations

MOB_BASE = {"melee": 1, "passive": 9, "ranged": 12}
N_MOB_SLOT = 20
N_PROJ_SLOT = 33

#: ``{tile_px: {name: (base64(zlib(uint8 bytes)), shape)}}``
TABLES = {
    10: {
        "armour": (
            "eNrtkjEOgCAMRbmAF/UyTMYDcR12BwdtaPIxpUWJJDrww9A8SPtsdG5kZMTFGEMIJUHAqfbeiyt67FNQ1HnHsA+moD+7ATIX5vxA5fBv4pirrs"
            "7yb+rzpC738J7/4c+8yljc2kk987a5/VAIDrIf07KKK3pMkA6KOu+Z5IMpuX9yyzBxac7fonH4N3HM1Vdn+Lf1eVCXe+jAP4p0u+PmfrScZqrh"
            "6Q==",
            (3, 4, 8, 8, 3),
        ),
        "armour_ench_a": (
            "eNpjYBjpgBGMYGxGDEkIDQXYdKPQOIQR5jAgmwiXYESiYRhdHIVGUYXVW0iGQxmMUBYjxHqo/pHtfwBz0gB3",
            (3, 4, 8, 8),
        ),
        "armour_ench_rgb": (
            "eNrtUssVhCAMtAi5W5BV0Y2NeN4mljY0mP1MhEF0ua28HPLIZDIZ6Lr73Offz+KdJs/+ALPM7gQtAcsUKUmYxLtflFeOPgWm5J7vxbeQkgJizC"
            "XDFVPmV98+93kewNTwo7DUH2TTJ2v8CS0/lZEox5+JjSa3tsdZwDNM4ZWMgclTzBAe9RsxsEyRUgxMpnDBNNbVRicjZ/fFLaSkgBjviVnDFVPm"
            "V9++91kewFTxg7DUHzNxe7K2/3/HT2UkyvFnYqPJd7bLrK1rBaM0wXM=",
            (3, 4, 8, 8, 3),
        ),
        "arrow_ench_a": ("eNpjYKAWYARC+usHAAIUAAU=", (3, 8, 8)),
        "arrow_ench_rgb": ("eNpjYBjO4H+TKJwcikBh5TM4OTgBAG0OBos=", (3, 8, 8, 3)),
        "arrow_icon": ("eNpjYICCw4cPM2ADuMT7ix2oIl4cY4hV/P///wxEAABZ0AxW", (8, 8, 3)),
        "block": (
            "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",
            (37, 10, 10, 3),
        ),
        "bow_textures": ("eNpjYBg+YHFHaH+xA5Cxf/9+uCBEhBhBoHZMQbhiNEGIYjRBiGJMQawqIQAA6NIcsw==", (2, 8, 8, 3)),
        "icons": (
            "eNpjYGD49+jRrytXGMDg15kzEMa/y5f/HTjwe8uW7xs2/N68+fvChU9qaoDifzdu/NvT87e5+U9z8++Ski8JCXdcXTdpaf3q7v4TH//Hx+ePh8"
            "d3G5sXBgbHFRVnCgsDtfy0tf2hrv5dQeGtlNQlEZF1fHwMMPBKUPAFN/d1Do4d7OwMqOAkM/NOJiYGUsCykqA1leHra6ORBbsTXDvjnaakeywp"
            "DlhXEwUXz3DVzfc2Kg0w70l0BmqECLpqi4eby8XZKKU4auZ6G8MV26kJeeuJ+xlIBprIQkT+g8GmTZusVYSQrbt8+TJQENNtQMVAKTyO33v0NV"
            "bxtTue1Uy4jiZ44MSbSYsexFReNI3bCxc8e/nDovVP89pv2Waclw1FiO869Kph6j3vwqsqcad5A7az2k+BiM9a8TCu5qpB0lnh0AOs7suYTJsg"
            "4oXtF22Sj0gF72T3WMFk3YtmNa/TDGaLDjTB5D3JGn3qmI43mm+IKRi6L1hoshCmuOlKY94mfpEpwsqzFTV70E0T6BPgLefD1GWz2JqngBdT3G"
            "mOA5HJJsHnES5xV727WBUAxZERXDEQQRhwvZiKcYljms9AGZDewyA8XwhTULSNCSgORxBxOAMoCyEhDDiAc3GJIwOgaZgmkAde1mliFc+05KeK"
            "OFWArRo3MkIWnBsrBkSd/nyY4kBBIIkpbirPBkREmoPV3suLc+EILo7LnfjB/ybRpyJYBP8fAyE0KSAXJN4kSqQ5QBFMQeoChZXPFHyeYRF8dg"
            "GEUKWAXJDgymfEmuODRZBUkHY/HI2BzAWSmOJAJHaYAYiQpSCCvCUMojNYgQxkcaAIUByIkMXhUmiCWB1DNmhxZEg2ZESpLAwZ040YEw0YkwwZ"
            "MaWA4nH6IHFMqUgdxjg9FHGg4UBDgMhUCsVGoCHJqIIQkIRNkBjwbFsoQXEgG4i+3l2DVRyCLs5QRxYHciGIRon/6cQbzyuv3Ss8CyThIk/KLj"
            "8oOn8oaMOllAN3809DpIDi9wrOXkzefzBw/TqL2Y9KLh4NQzRjgGru5J9+XHppm/Oi27mn0GzZ5bkCaOZ216WYDgCawzCsAQDV0J+B",
            (15, 8, 8, 3),
        ),
        "item_a": ("eNpjYKA3YGQkn0W0FVAtjIxQJiMjjEksC6EXpxXospSI0dMfAGJWAJU=", (5, 10, 10)),
        "item_rgb": (
            "eNrNkDsOgkAURafUwhg7GxdAY28HyncYoHULbsIVuScrC1eBxxkhEwgKUdGbl8nknnfffIS4qxRlo0SlNqobutA/1/MXiVe/0VMXZ3E9Ol30fF"
            "pTXfSwmVPj0x9qptUw96uJ3eAvpw1qCt+UPcHOoja1g9/LflDt9w7N2nceLWsHB2XfVKFkkUvWNE0bKEmSKIq2WkEQSClrRDMOvud5ruuyp1kp"
            "ZehOi2wcxzSEYcimpqSgmKYHEa8pc3wtTLIcipNnD8occy6rmclNsooi00+csUqrzyfcACQO/+4=",
            (5, 10, 10, 3),
        ),
        "mob_a": (
            "eNq9VNESwzAIkv//6e3aBtGwptvdkpeyLhUVMWLPATrCcQrCAd+PBYrz55rDIkYBNIdIZGIbsuu6oMHBuMIAjZy5kGtmhWZMgoYKP4Q4Bjrjwv"
            "fKZt/rOBs2nqpC1QOje3ikjNZJ9WU2UiNyRHLs12NPHVZ9Jm0qoWiCvnYju4F2JI9fjY+mEA0PobO1qUFNSLGsa1BUa1/vgmOSm+jJxBp/gLNW"
            "1XDf3qkwpb9C1oNu1iQbqBdaRfqy7GJtOHrtAbtZa+S8P0st2eNul2DDmebTrJfeQPn3w8WFB/7A8QKTCQMM",
            (20, 10, 10),
        ),
        "mob_rgb": (
            "eNrtl09IVHEQx5+GgZGwFpEVZEakhxSD6tDBkmLZsA4lLoLmSqFgtMhiCREFQUSnLPEilBB1CYRuFWFUYCBFh8DDnqLLduhWdF+/b7/uOP3+PH"
            "e3jA4Nw2PefOY7v9nH7mMnCP5bJTbx5v6B8biPAjkpVA3H95AytlVIsgYFWiiHsga3IhdKrY9qrYzHwKfVHex8ifauvX3eT+cL7rOfnx9/X7zl"
            "o0A+um4wyOfzuVyu+kR4Ozw8rHuCJoOLcFCgnn113d3dWp4I+uAUdjUFPgoD/Ts2NTW1pTeAI9D5ZDKJ6/Z0QAq3KcxGpFqoa9LpNKiBdIee9p"
            "jd2T4Cz8f3iCTvLMBgvp4RFhsIfeOxKidFnm4kGTQ+rKE76d5ntUB2Z2ZwpdbZnIeS0vWQMpJQPeT65sAIjLgUSxfMSHZ0dBANFM2msI9P++FO"
            "LcwWkg78aolEYjIW5pMH60621YrQoMiT7laG28mY+XH0zDaV2fSoxnNwTr6qNQyGXjH1WeO1Fa0uQ17aOuUi3JQIHcarnsdH7ZE0ZWc7X/r76s"
            "vIAtz5vsrd+EQKd76vbESqhbqG7ysD6Q7yvtJ5+4iK31cYzNezFKvvugyvmPrQrpEZ0eoy5KWtU74ibImHDmuJG/P4qGMkRdnZzv+mpY80w7PZ"
            "rI9mCzZ94QydsVEgTRg/uJQy5AhSh5rgjEntQ3VGypx0rW1zwcqV+JKltyr30H/W4q1VdCc931kTUaCp1Cw+vwd3UrhQZ4FBjQKhDAx/OXOlrA"
            "++UDRzixk7SoprKpUyEP51A00XTctJmYdQakiRAUWAzOjoqOS1ludKWzl9rH8/KTPsb8/MJLQGlQGc2jXacR7d7onYcTiwc8eBkB/f2HGE4iMY"
            "O450wINybjG2OamWV2zVwSm4441RP4Q8twnW4OutVXpVwW3v4Rqjp9ZqCiGbS5mmuPVR7iCa2vuIQXVnjWyqkU0zmYzEzn+qmYKNW4/x2+wygt"
            "kqUJE7//1qrfNQJPXVGJgqPbwtF5XRX+dt5FNF7zjOfcH+lTl3nIjnj1+Z7mlsMdLWt+PQfDsO5X9+xzk923r1deedr3TEdOThyOCK98m2uztZ"
            "yVhqUIBb0g2JCYmlJzOsR1B3rl6fIhlWIpbOQjmJUB6E+uaxD1IvfYCopXN+XNmHBWH90KsdZ5+wAFe04imi0kNKLA/zR9HnXF8MQXNlvgO1cM"
            "5DK+uc71u9gP4+EbpJ3y67WxtJ266/wH97erk0wiCM0GpUbuetyZulLGXeLYm7j2+FcdElPc0U9g==",
            (20, 10, 10, 3),
        ),
        "number_textures": (
            "eNpjYCAZ/AcDZC4m+z8MoLGRtWMK4lePqRerS3ApwGMXmu8Iug2XdZjqsWrBGmiYfscjjlUN1rAFABt76Rc=",
            (10, 4, 4, 3),
        ),
        "number_textures_a": ("eNpjYEADjIwgAoIZgRwQZgDTqHyIGFQ1jAlXwwARRuIzIvEZYGoRmhE0lA21EwAWfABP", (10, 4, 4)),
        "number_textures_with_zero": (
            "eNqNkEkKADAMAv3/p1vwEKQaqacgTjYAhwILUOOM+ThTK+UdlHXczZ53Nm6yBcosiH5228Z5PiLxaX578WMm/vYCvyUBDg==",
            (10, 4, 4, 3),
        ),
        "number_textures_with_zero_a": (
            "eNpdjUEOADAIwtr/f3ozQ012QBERUQQLr14QWMuIfvPT4m46Htg8O31m2rvH28Pz8wAbQABX",
            (10, 4, 4),
        ),
        "pickaxe_textures": (
            "eNpjYBgOYHFHKARhlS2OMcQqBRTElIIIAhn9xQ6YghDTMAWxqiQeTIUBrLKJiYlYpSB2oUnBHVBRUYHVVUDTqO7+NzCAVba/vx+rFMQuNCm4Aw"
            "4fPozVVUDTqO7+rO/fIQirrND1W1ilIHahSSEc8Ps/VlcBTaOi+wGMOm3b",
            (5, 8, 8, 3),
        ),
        "player_a": (
            "eNqdkUEKACAIBGf//+lA0sokIU8T1Y4WgMQsJ1kFaIE2oiJsOWOCXo5NUmrybu/wU5cDT0Edfc/RO3ReST1XiUofwuut2jkG958BAA==",
            (5, 10, 10),
        ),
        "player_rgb": (
            "eNrNlDEKAjEQRecUVtb2HsB6T2CrnbZ7AMELxEawEyzFQvACWlpaWG3nHayF+Je/zM4mbgoRNQxLyEvYycskInUbDnoIaWm3bY6Ix/djmV+K7u"
            "aAQKeTzxQ9/BqUyAaR9421XMivd5KgnAB63q2WkwwxzfrMkPRenDABFOMMDB4XI5s2KDukQVMPbULec4XtWFfYgrbYEigs6UJ+1QYoFHFCG4UE"
            "2uB+1VVJnagrUnUF9FlX5Vk7aUW+TjWoqyASrrSuXtqwlZOmsaugrgJX/1lX9g7+0FVwfHFdJU6fKlg5CVeYYCNwVT4jrnpM9BfWVZWAq/Oxik"
            "p0FX2veGG5FuONcGJpXLe08eX36gmaxZ4W",
            (5, 10, 10, 3),
        ),
        "potion_textures": (
            "eNrNjiEOwkAQRSepqMA15Qg4DjDhoiDRiLVoFEGtLAKHQ3EEXjPJZDICKhD788Tu/9v0ifyIqtZal/RcNSS+JK+us4N/RWOc+h685+ATpF9YZH"
            "H+5T/exuRPY8hBIPr7NK9t+G92z+RPY8hwhOjv07y24X/Z3pM/jbGXAtHfJ2jEf/1+JH8aY3U9Q/T3CRrxnyZN/jRGKQNEf5/gu/8HgvP8Uw==",
            (6, 8, 8, 3),
        ),
        "proj_a": (
            "eNrtlcESgCAIRNn//+maaRIo0wRqOOCJCwvPBSX6+wCdiJYjmMsfp0nAz+HoPoLDJyQ1hx4hsPsbCSN9NGHj7hGI4eRomT09FhYlZIbd8T0XPF"
            "cqUsUsF/PMNqFcmLBzm/U0XUCk9EuU8iOpH/V/pPg/aj/qvSo/yo+8fmycqwJ5",
            (33, 10, 10),
        ),
        "proj_rgb": (
            "eNrtWEFOw0AM7JkDJ6JKnPoFpAjELQ/oEyrxiP6gH8iVL+QdkTgj8QbyBm7FYYkT1utpAizbRbZ8SDvejWc8uXi1svj9aNu2rmsN3e9uqEBD63"
            "0Fbo6HUlcAPX5ELuITFyLLST+5eZqLFD8XangKeIIJfQXcjr+UcxNf+oqpyblkRO1kVFUFoLIsY6DbzeXd+uJ7aI4KuyBelFIlF0T537D+yXzj"
            "uQ67PTuFKdlaHuq8xNbKiPXrVZ8adHwqKGVBDx0Kh9JDuEA5Ox4/FODVGNLQhDICyDF1NdNKVollnI/Kmxd1ldxyWDHkSd058RyLu0piMLYT20"
            "B2O1L+ygujXg2e1B/HZttRqlD33Kco6KFmQJsuXKCcHY838NUQ0tBUAfr5JDv0PK0cVRpknI/Kmxd1lVYrPMET09edE8+xuKskBmM7sQ0CIjcT"
            "C3nW0lG/Bk7K9le2v7L9le2vbH8VjOvHNaUGPbztKGUB/XP/cutQeggWaGf5ePDgnK4AmlBGADmmrmZaySqxjPNRefOirpJbDisGGAHnxHMs7i"
            "qJwdhObAPZLVP2eGHUq8GTsv2V7a9sf2X7K9tf2f7K9le2v7L91Rnur94BSTVM4g==",
            (33, 10, 10, 3),
        ),
        "small_block": (
            "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",
            (37, 8, 8, 3),
        ),
        "sword_ench_a": ("eNpjYKAcMDJioRgY6aIfAAVCAA8=", (3, 8, 8)),
        "sword_ench_rgb": ("eNpjYBg+4KkIw/9jomQLQsQHECj4PFN4doFsQYg4nd0MAA6wFD0=", (3, 8, 8, 3)),
        "sword_textures": (
            "eNpjYBhuYHFHKKZgf7EDEOEXhLAxVRbHGGLVDrQIIoUpCNdIhvunTp2KKVgBBvgFIWxMlYmJiVi1Q5yKJg60He5soEYy3P/mzRtMwcNggF8Qws"
            "ZU2d/fj1U7xP1o4kDb4e4HaiTD/Vnfv2MR/f0fhPALQtgYKoWu38KqHeh+qBSq7SBBuEZCAAAAy3Cl",
            (5, 8, 8, 3),
        ),
    },
    16: {
        "armour": (
            "eNrtlDEOgzAMRXOBXrSXyVT1QFynOwMDpURybCc2RkoEFD95gJ8nRbEDITiO4zjOn/NZGRClMNTYFKpOjJE9gJPCRPLhFbS9DqM83bUmBY3FHZ"
            "BmAUdmDg4Zy5LdUeZucfDcFdo6itZ1XtJ2PRx7gsNzOvjjxReYJTf/gVe7eohjuZlteY7jUmGacxUCWS00Uag5j9ebPeSd1jBV8uE1azsdVuXp"
            "LgTuM+6ANIt8ZOqQkNZvyewoc7c4eO5atXUUree8pO26OOaEhKd0yMeLLjBL7ozY1UMcw83c5Asw2WVa",
            (3, 4, 12, 12, 3),
        ),
        "armour_ench_a": (
            "eNrtkkkKACAMA53/f1pBSl2KWHJtLkYIddS0ViqJAjO+EAY3TzDh9GNn+mUJ4vlzSXNGECvPA+j0M7s+quYdRf5obAzft7ktNofqT/VH6k8H/s"
            "cA4Q==",
            (3, 4, 12, 12),
        ),
        "armour_ench_rgb": (
            "eNrtVNsNgzAMZIjyz0BMxTZdhO8u0azRhtgY8rB8kpFopVh8JOHk+O4cD0OPHj169Ojx9/FZx/hpvxqHy3H4fviuXkakPAS5sbAwsVqzYALEVD"
            "Eh4SkzbbcFUPP1HqV7uYDkF21B/TV/ay9snQEFEC/MPOz7anuK+I7UAzYkwbg3xI6cMrfN3jnFArkCcRbB1ALW713Lcy4VGREatdswewuJ/oUI"
            "zvH1E6P7xEuGlbNDPAPEk0TeoDyWgleTzhRe8WsmbJ5Pz3Cs5+DR/5xKxaQaTOTGwsTMwSyYADHVljDhKTNvnwGp+fKge7mA5BdvMf01f2svbA"
            "0BBSAvrDzsu1U5aXIBBqN/FD/nduSUuW2kc/IFdAXgLISpBKzfu5YnYwSMCI3abZi9hUT/QgTn+Lo9Cl4yrJwd4hkgniTyBuWxFLyIzhc7W2tv",
            (3, 4, 12, 12, 3),
        ),
        "arrow_ench_a": ("eNpjYBiEgBErEycbnTec3QMACQIACw==", (3, 12, 12)),
        "arrow_ench_rgb": (
            "eNpjYBgFQxj8PyZKQEGTKGFDqKQGAp6KjEYLTqDw7AIBBSufETaESmqgKn2ejcYLHAAAhacPyw==",
            (3, 12, 12, 3),
        ),
        "arrow_icon": ("eNpjYCAZHD58uL+/n3I1xTGGQGX41fQXOxB0Dz3VAN1MUM1/MCBGGelhzwAAn28aNQ==", (12, 12, 3)),
        "block": (
            "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"
            "U0tvmUzG6LCdO3dO9GzGrHilNotmWLPdOZXLy0X7SpKLHb8zBScTZdOIQW6lY9c/Pq+YasdEG2UhxmlUfezEkyyBc9EzVjP2QEwMzgjlVv9j8f"
            "2mxn+tr3utIf1Ge9t/9fa8vmvXG5nMf6brX9q0yfjX5Jnye15oSv9Te/NLfd0/2T/82qED/dUbftrZfmXvntf39v97S/O/NW8X/uk//P3LzY0v"
            "7sw+OlD77b7MXzVVfitTi/5Te9qea67bVHzjT9J1P2tsuJKqeaWs7PVU9T93tFza0XyprZG/dPHc7vZn2htJ/nXVPXD0fH/3TysrXk+lflyxda"
            "6q8nutDV9rb3pyRwN/OeS1VM38qVOAv9uz87+PHwP/RHfr5aH+Vzs7Xi4v/1ZjCkN/lqn8cnP1V2tLwf+wtMT5+ycvt2x/pTF9pSn9fGMNNcfr"
            "Su3fP3l+e+2rOzvn6u77War651Xb3njgKPRP9WX5syF/3lYj5GOVd32+8k75/ZNvVpZfqqlEiV7YP/DS8PAP+rqf7dj+le0V8vsnP9jT+42W9N"
            "+na16su+8/Robn6muNt45Xj0y+MjWhf//kR6Ojc2Nj8e8/+vdPkrz/6N8/SfL+o3//JDnPwuZ9/7F//2QFvf+UNXQu9f0HU5b6/oMpuB06+Xla"
            "+MLjX6EmKh7cdu7/DDqYMv7Q2c0tfc73n/TB0zU7R4TH3D1H/2jv1HTM+8+23k/vHDsBzcDUw20Dk9nBybzvP1s6hjDEwGq7Bjk6OvuljsNnek"
            "/8ifATM+f6j32OKTTfP9k2PoP+wLE/ru0ezY5/Nur9p2XMj7a6a2Rrprd8+25JP+r9p77vUGr3gY7R41ta+ypae5O8/3SOPtA5eiw7cmTHvqOo"
            "UvPwA7/i7z8r4vdP9m2ZGSh9OMnvnwDbv3V6f8Ujg/dMj1U9gk7M97/DZQ+D2Xevz6MzXvMoHA1vnrY///GKrofs9b7uFI7qz3+o8YVNbpXoz3"
            "/8P4P6x/dBCIf9AJO+/vyHejHlM4dybnHVn/9onvCP13oQRii8fP5j2A8TWUiHQ/rzHwevhEP68x+TcVXp6nz+s7r/f0X2/wfnvgMhHPYDTPrG"
            "/hcgZC7M6Vt7/+cYvDBHYYTC6/2v7Rv7n0PG/jd5JRxaKfu/tuCDd3s3UPLuf8Cah0Bfs+ejDQOfhNT0f0Lvf5L/8+JjV54ep3AKMG/0Bu/eot"
            "bhjXr/gwec8jzC7IBP3/9J8vX9H9X73zAuLhDJp1qvR1RNA5/Q+5/eL/1lHUl0qEFBajdc6zz/g9H5+XmQEHSYr1+HDdfa538QDFKYVw08HAEA"
            "b3z+z8hZIoF/y1vD2MjL/pc0pfjgAQsP0fufPCwITxiCxRXe/vzf5vkQQzzOz/9tnp//g7c//7dhCH//ivU09r+T5/5nffT+d8IQ7v8wnpXw/C"
            "9VLe/zH0w6ne7o6NixY0dXVxduY57/GN0etPr6+lQqVVFRgT6UzvNv0Le0tLS3t8M4LFdXV1dWVvIWQ/b5NyibmppaW1sB1NXV4QoYmmw2iyH7"
            "+y8oMZTJZJqbmzuDhhSgwS2G7O+/oGxra2toaNi6dSvMIja6YNb2919QIuBt2/zvs2pqanCFL7iAIwzZ338xHn45hQ4qAxLGYYS8sf+ZL2CkDB"
            "6R1NbWIjb4wpD9/RfriTIiQcCwv2XLFoSHqDBkf/8FJYwjZRhH1jCOK2A4xZD9/IeSJ/dYSW4J9BEehlbQ+49s/oKxDyxp/4Nf0v73qouS73/A"
            "b8msTb7/Cf/a73+kCUm4/0GijMb+h5Jrbe9/wNAb+588rnr/4xYkr5hl7H/6XUHv/yv69z9Ref37n/7vgUeff+b+l9//jNn/cv6B+58HFWL2Pz"
            "zq/U+nq89/e/8fL/jNs977KXn3P2DNQ6A/13zzn7Z+DHKuuVjvf5L/+8ThNx7fTeEUYI90vv+RO959IXub3v/gAX/ReythdsCfz3yM/Pnmm/X+"
            "N4yLC0TyWFVREFWx3v/0PjezmSQ61KAgX7zlOuf+x+j8Cy+AhKDDfFEH8Pb+RzBIAZgIeDgCAN7Y/4ycJRJ41lsT5h7wsv8lTSk+eMDCQ/T+Jw"
            "8LwhOeDd7/hbff/22eux3xON//bZ7v/+Dt938bnl14/2c9jfd/J8/3f9ZH738nPLvw/h/G0377/wFIGHU3",
            (37, 16, 16, 3),
        ),
        "bow_textures": (
            "eNpjYBgFwwH0FzvAEZrU4o5QIIKL79+/f/v6FVi1w7mYCpBNgBhCUAGaIWgKyLAC4k48JqCpwWoCxCI8QQFXg0cWrga/AiAAAFj3PfY=",
            (2, 12, 12, 3),
        ),
        "icons": (
            "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"
            "AOJqegMawC61Bvs/XtP4A08VvqU=",
            (15, 12, 12, 3),
        ),
        "item_a": (
            "eNpjYBgF2AAjGneQ8ekcGGCAykMIMKIKMDKiCDDi4zOSzkc3nwGNi+k+dPfjjGpGHHx084eY+tH4xxP/AE6XAXA=",
            (5, 16, 16),
        ),
        "item_rgb": (
            "eNrtkj9OwzAUhzPCgBAbCwfows5CAiRt4zhpEQxcgY0TMHEc7sTEwCnKlzxiqtiOnJQqEupPT5b1/L3nn/9E0a820cYZkSUf6awKgbdLBvHh8C"
            "H2Eft+3xH/bcR/3l2fs7NB8NfrLJwH/ni/DOeBB/HPV6fEgT9oR729VIHkSase5uniyC5Jz48DeSmB9+0CbwLMhM9YT/9AHhneLnGa6eeH9h/h"
            "p2NG7n/QecPfa0L1mHEeYQTv+29/1X+0H+cTT+XHNjOhH6eZSfz8D620qqOqx0Kpoih8pNaaVaXUcrmcz+d3jRaLha+EfJ7nAFmW3bZiTod1pW"
            "3YkDeNkiQRnialrrXtGYAlDMCoxjYjSYxhT1s8ZJqmYliOILDwUvJ4rw0vW8MDyIRC5mJS+If1Dw8srWQLAEOyhH/y8Kv21HkjVtNWzPFDLd5Y"
            "on9VasIcgYxpKFuQ4dTyKKgstX35MDSXy7xuRFKqnO8ru8dxTJW8bOcmnVUd7fghvwHih3mS",
            (5, 16, 16, 3),
        ),
        "mob_a": (
            "eNrtV0FuwzAMI///6fWQ2iIl2UkzYAMaYyvCJnEtipJp4BkAX+MKhkAew+DxBalfvD4P8Dv4WE1c0Q5XBITrjGP8NIwT2OYb66cTxBkgBkaNx/"
            "vvPxCK5+/t4jc2xsd79TOghJkxRA3pvqohySPEr3zHHCz1eV8Prm4Xe41bblw7C3IQ2KDgIQ9nELMyDPN6/Z/J7y7fnipqTD6g12gwwm86futK"
            "89v0h1g8p/lZSKKYLD8eGDnep7QLzvIdocfbnDXuOHaG9A+7nil76v/J/32vwApruyoyHEPMGLK/W2vZ4r/1R5zsWmtbDKmS+b7p6N9aRfZiF3"
            "dI2P4mmEh6ILsW0VjM/Hh6eB/V2vDRyodoF0exbsOHzfkMK0nh96RhOS4M3uWGl/oXb9EjfoJao5OO9nSAWf5CDxp6pFL4gf8tNZwVbGvOGCWW"
            "ys73N+ouzzsyXzN/h5mwt1FWBDSnvYy3+2UzgRmEsa6mx0D8nfJjKd3uf8v8E3a8NT+z1XtjdUPX6zFo50VpJ5R2MM6DF/wvv3xsz3QVxn4zMf"
            "WJJzy1G92ZwFfrjrTcw78z/h96NQfh",
            (20, 16, 16),
        ),
        "mob_rgb": (
            "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",
            (20, 16, 16, 3),
        ),
        "number_textures": (
            "eNq9kjEOACAIA/3/p3XSKBwVHGQSUlsKtPY7+gyTrspez1f6GVjBBpA8k0Za6PeqFWFQSyB3QESCUZoqLrTkXe/dW4i0zBsHoq+uenilFNv2PQ"
            "trz7/EeWR4DJUBDFbEBRg=",
            (10, 6, 6, 3),
        ),
        "number_textures_a": (
            "eNqNj8EOACAIQnn//9NtRWrWIQ9sIiJKPwVMYDUnMgd4XEUVtkbNIvXWRCeSdrWLjtY8MyflujdtnR9x5TnAy2s3jB5MhK684usBbRAArg==",
            (10, 6, 6),
        ),
        "number_textures_with_zero": (
            "eNqdU0kOACEI6/8/7SQejNoldnqSBiggABgbMHE9lrlIZjgKKvNOBh+nEphxQjKyAJn8xXRaOPGo5XykVvDk+XMSiWqqXEDbe/53bsFp8d7KUs"
            "PWtYtXmbJsd2v5NNqosB4/TvVy+AAZjDTo",
            (10, 6, 6, 3),
        ),
        "number_textures_with_zero_a": (
            "eNptUAkOACAIgv9/OqekdLiGJXgkGAbkCQhXCAWTRWlIVzQyCbpUtMPW4Coxemn6Vf0rLLs6AvjUnDlp3ZWp0vMjPvMcoGT6Or6RHvpZXF4Xgn"
            "gAvg==",
            (10, 6, 6),
        ),
        "pickaxe_textures": (
            "eNpjYBgFwxwUxxj2Fzss7ggFIsqVAdVAlOFXAzENqyxQL1wKqAyrURATkLXgMgFZC5oCZBOwugGPAojbBiS+EhMTKyoqpoIB5cqA3oQoIxhfQG"
            "UE4wtoKVaj0AITTQ3W+EK2bkjHV39//+HDh9+AARWUFTtAlBGML6AygvEFtBSrUWiBiaYGa3whWzek40vo+i2G3/+zvn8HIsqVgQIKrIxgfAGV"
            "EYwvoKVYjUILTDQ1WOML2brBE18AfSm9EA==",
            (5, 12, 12, 3),
        ),
        "player_a": (
            "eNrNk9sKwDAIQ5P//+mt3U2tVkpBlrcjQo2xQBNPQShjSOQjg1eBVAUJHsMyMr6n0ROt8Z7/tQWgD642MmHTD4cL/KvHPkcBa/vgsA+YPPf4P/"
            "lX+WdwcFFeKYPhCfv8ft254cF/eu+szv8ABhsCdQ==",
            (5, 16, 16),
        ),
        "player_rgb": (
            "eNrtlj1OAzEQhX0KKmp6DkCdE9BCB20OgMQFTINEh0SJKJC4AJSUFKnS5Q7USOat3mpizc+uXaQIxBpFq+SblbPz5cUp6XV+diKVGtb3+mPzvE"
            "S9313sgsdqgUtJr5fp9muNOn56Q/H6aHlj4Z/ySJ6kKrTgVcFy/6hFwZaXzdS7Kjl18XUL+PurRV3Xi1PLSwv4z5cHgXFNnhNRMxpaciLDqvlo"
            "HMKo62jtmq81blR6T/3n3K3/GDpGrxY/dU0mD43VnSm57SKvWmp1lc/C1y2NPISU/ZO3/m/5rJ85nVH+j3zWM+IDt/7zd/G3/R9CLKdG/7eJl3"
            "vghhYNz7VglG7ZNeG/zf9p/1X+z/rs5nMvP+G/y0/4H+V/5P8h//cr/93zz8H/Fv/dMEH+RP47YdUbcXMRij3Xedvov+R5u/9ocSvy2c/qVeg/"
            "/0ZZY+8q/Pq1/xGMN63/EYz3rck1rM7/1n//+5Zxe3J0EX64rVs5ufzEfxm3R77L/67zPx74fzj//wIhuAhq",
            (5, 16, 16, 3),
        ),
        "potion_textures": (
            "eNrtkSEOAjEQRZusWIHbLEfAcYANFwWJRtSiUQSFXAQOh+II/N2/TCZMaZjgSJsnJjNvmqY/BN85q5PRuvHkr/rRwQM6c/SrrHCvqrcL4aAp7O"
            "qaoHY5WuOIjv0ldGQdfPrGL//5v/NqT63NC00hbAIZao+jtWk0Osm8ZB2UvDLCYnWzeaEphGZLhtrjaG0ajU46r9c6KHllhMPyYvNCU1iHSFC7"
            "HK1xRCeZl6yDkldGmD+uNi80hdlxT1C7HK1xRCeZl6yDkldG6PvO5oWmEGNDULscrXFEJ5mXrANvXk+f8BH3",
            (6, 12, 12, 3),
        ),
        "proj_a": (
            "eNrtl+EOwyAIhL33f+klyzIrqNR0a+n2+avUaoC7g1IKq7NkTGnJtue/bYf+uQtOTOXbl+eTtquznwT+m+Md+XcZ/I4AGeFfLxAeoCN2Fcfe8/"
            "dKkJoAm2qwWxHJov0wniHeudMz987C7QmQLjpfXzXG14ajfsATfni6tNs9fihVdlRB00DurzfWHqRrywJl4IMmeET4zfFf5k/x1VMXckAVsl4A"
            "IQHs98Xed36E6B/9o///0T/zP/M/8z/zP/2f/k//5/8f/aN/9I/+0T/6R/8/pP8Hay4FUQ==",
            (33, 16, 16),
        ),
        "proj_rgb": (
            "eNrtWkFO40AQzJkDJxASJ76AFIH25gfwBKR9RH6wH/B1v+B3ROKMtG9YTnnA3mDIkNm2E4+rxhk7Q6rU4oCqzMx0dTeHXiwEQRiF9XqNkzebje"
            "PXdZ2J78ir53vqSPWqou57bnz3njj5fYd8/PBTmAabLfpqqpPB9Q7ONs5pnXC/3E93vQX4feGkwNZj7v7A9iv1/+PO09zznf1/Qxhf4JECdLnw"
            "02Ff0tf/w4AAv++ngxJxzqiqKh/fkZfLZYn8p7vLx5sLG7gKPw/LF0qEyzLFRyxRGTi++xN9f8VzlgaUpcHzjOELBTk5U39L67flzpdp5q/wzd"
            "p++LKvFzsF4l/2YacAolLbLwt/rz4DJ7//unaBSL7IL9sYkrTIgITls+exl6WvjJHD44Mp8BwqX5ZPqYQZKyuo+uQdJ3dUSKUEyb6K5bPnidw3"
            "7lLW+UJWVyP9M6EK8H6YOI/gfl76/EqYR6qyuWaB7flg1iwz7opQUy1yv4TlJ5znoEounR53T28uCHKzDUDyRX57/YwhSYsMSFg+e57WZdkrN+"
            "iT+scHU+A5VL4sn1IJM1ZWUPXJu05uYhY6XClNr/FYPnueyH3jLmWdL2R1NdI/U6oA7oeJ8wju56XPr4R5pCqbaxbYng9mrcWMuuJ/WTWQ61h+"
            "wnkOq+TS0dD+//fma///zKH9f+FY9aj9/7L42v9XgWv/Xzgibn/fuMDJP/48uEAknvzz37OLQYklIxKWz57HXpa9MkgOjw+mwHOofFk+pRJmrK"
            "yg6pN3nNxRIZUSJPsqls+eJ3LfuEtZ5wtZXY30z4QqwPth2jzC+3np8ythHqnK5poFtueDWbPMuCtCTVlyRMLyE85zUCWXTg/t/w9eVvv/wpSV"
            "pf3/8e8p5Ha19v9PfH4ttP9fzizQ/r9cOgu0/z98We3/CxNWlvb/x7+nkNvV2v8/8fm10P5/ObNA+/9y6QfR667C",
            (33, 16, 16, 3),
        ),
        "small_block": (
            "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",
            (37, 12, 12, 3),
        ),
        "sword_ench_a": ("eNpjYBiUgJGREZPFAGciCzMgCzMwDnP3AAAXcAAf", (3, 12, 12)),
        "sword_ench_rgb": (
            "eNpjYBgFQx48FWH43yQKRLhkQQqO4ZSFm0DYimOE1OBVALduhAMFn2cKK0EIpywQPbuASxZuAmErcBiCUINXAdy6kQMAqVsqfQ==",
            (3, 12, 12, 3),
        ),
        "sword_textures": (
            "eNrtlLENwyAQRZkhm2SDLJEuDW1aN4yQjooJMgpDRMoM7l24cD5CXCi4b9PQJF8UJ/7XuzMn2Zi/fkXPx9VPl+l25gGeyQGcLghq3MASQrOFWE"
            "da7H4I4dQELUP4YxRCcM5Za3mAZ1xRFwQ1bmAJodlCLN4Cr82HlHVonHpfWobwx2ie5xij954HeCYWdUFQ4waWEJotxOIt8Np8SFmHxqn3pWUI"
            "f4zuy2LW7fR68wDP5EA6PZBUr1uyCqHdQizaIv1X6ZB5HYSDSb5/VCXDHuGwPm5d/jQ=",
            (5, 12, 12, 3),
        ),
    },
}
