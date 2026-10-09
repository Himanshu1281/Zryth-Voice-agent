"""What the caller said: every keyword pattern that classifies a caller's words.

One place for the yes / no / goodbye / booking / callback / office-hours / refusal
patterns that used to be spread over agent.py, contact_flow.py and tools.py. Each
pattern documents the real call that shaped it; tests/unit covers them.
"""

from __future__ import annotations

import re


# ---- Answers: yes / no / which number ------------------------------------------
YES = re.compile(
    r"\b(?:yes|yeah|yep|yup|correct|right|sure|ok|okay|perfect|absolutely|haan|han|ha|ji|jee|sahi|theek)\b"
    r"|हाँ|हां|हा\b|जी|सही|ठीक|बिल्कुल|बिलकुल|ओके|करेक्ट",
    re.I,
)

# "ना" alone is a no, but in "सुनाओ ना" / "बताइए ना" it's just a soft "please".
NO = re.compile(
    r"\b(?:no|nope|nah|wrong|incorrect|nahi|nahin|galat)\b|नहीं|नही|^\W*ना(?:\W|$)|गलत|ग़लत", re.I
)

# Short acknowledgements that, after a sign-off, mean "okay, bye".
ACK = re.compile(
    r"^\W*(?:ok|okay|ok ok|alright|sure|fine|great|yes|ji|ठीक है|ठीक|अच्छा|ओके|हाँ|जी)"
    r"(?:\s+(?:hai|bhai|ji|sir|madam|भाई|जी|सर|मैडम|है))*\W*$",
    re.I,
)

ACCEPT = re.compile(
    r"\b(?:interested|sounds good|go ahead|let's do it|lets do it|why not|please do|sure|jarur|jaroor|zaroor|zarur|bilkul|"
    r"karo|kar do|kar dijiye|kariye|karie|kijiye|book karo|book kar do)\b"
    r"|ज़रूर|जरूर|चलिए|कर दीजिए|कर दो|करो|करिए|कीजिए|बिल्कुल|बिलकुल",
    re.I,
)

NOT_A_YES = re.compile(
    r"\b(?:thanks|thank you|bye|later|not now|maybe)\b|धन्यवाद|शुक्रिया|बाय|बाद में|अभी नहीं|फिर कभी",
    re.I,
)

SAME = re.compile(
    r"\b(?:same|this number|this one|current|calling from|this|here|callback)\b|"
    r"यही|इसी|इस\s*(?:नंबर|नम्बर|number)|जिससे|इसपे|इस\s*पे|इसी\s*पे|इसी\s*पर|कॉल\s*बैक",
    re.I,
)

DIFFERENT = re.compile(
    r"\b(?:different|another|other|new|second)\b|दूसर|अलग|नया|नए|और\s*(?:नंबर|नम्बर|number)",
    re.I,
)


# ---- Name and number steps (contact_flow.py) -----------------------------------
# Caller indicates they already stated their name ("bola to sahi", "already told you")
ALREADY_TOLD = re.compile(
    r"\b(?:already (?:told|said|gave)|told you|said it|i told you|already|said)\b|"
    r"बोला\s*त[ोॉ]\s*सही|bola\s*th?o\s*sahi|बोल\s*तो\s*दिया|bol\s*to\s*diya|"
    r"पहले\s*(?:ही\s*)?बता(?:या| दिया| चुके)|pehle\s*hi\s*bata|"
    r"abhi\s*t?h?o\s*(?:bola|bataya|kaha)|bata(?:ya)?\s*(?:to|na|tha)\b|bola\s*(?:to|na|tha)\b|"
    r"अभी\s*तो\s*(?:बोला|बताया|कहा)|बताया\s*(?:तो|ना|था)|बोला\s*(?:ना|था)",
    re.I,
)

# Questions, goodbyes and fillers: a reply containing these is not a name.
NOT_A_NAME = re.compile(
    r"\b(?:what'?s?|who|how|why|when|where|which|is|are|does|do|can|could|tell|about|"
    r"bye|goodbye|thanks|thank|fine|okay|bot|robot|lol|kidding|price|cost)\b|"
    r"क्या|कौन|कैसे|क्यों|कब|कहाँ|बताइए|बताओ|धन्यवाद|शुक्रिया|बाय",
    re.I,
)

REFUSE = re.compile(
    r"\b(?:(?:don'?t|do not) want to (?:give|share|tell)|not (?:giving|sharing)|no need|skip (?:it|that)|"
    r"leave it|forget it|stop asking|again and again|not interested)\b"
    r"|नहीं बताना|नहीं बताऊँगा|नहीं बताऊंगा|नहीं बताऊँगी|नहीं देना|रहने दो|रहने दीजिए|छोड़ो|छोड़िए|"
    r"बार बार|बार-बार|ज़रूरत नहीं|जरूरत नहीं",
    re.I,
)

# A preferred date/time ("tomorrow 8 am", "kal shaam 5 baje"), not phone digits.
TIME_OR_DATE = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|baje|o'?clock)|tomorrow|today|tonight|"
    r"morning|evening|afternoon|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"kal|parso|aaj|subah|shaam|dopahar)\b|बजे|कल|परसों|आज|सुबह|शाम|दोपहर",
    re.I,
)


# ---- Wanting to be contacted: bookings, callbacks, a person --------------------
BOOKING_WORDS = re.compile(
    # Generic across businesses; each business adds its own words via BusinessProfile.booking_words.
    r"\b(?:demo|consultation|meeting|appointment|reservation|booking|slot|site visit|trial class|demo class|"
    # clinics / hospitals: "cardiologist ko dikhana hai", "doctor se milna hai"
    r"doctor|dr|opd|specialist|\w+ologist|checkup|check-up)\b"
    r"|डेमो|कंसल्टेशन|मीटिंग|अपॉइंटमेंट|बुकिंग|रिज़र्वेशन|रिजर्वेशन|साइट विज़िट|बुक|\bbook|डॉक्टर|ओपीडी|चेकअप",
    re.I,
)

CALLBACK_WORDS = re.compile(
    r"\b(?:call ?back|call me|contact me|reach me|"
    r"(?:contact|talk|speak|connect)(?: with| to)? (?:the |your )?team|"
    # "उसका नंबर लिखो" / "take my number": the LLM said "नंबर नोट कर लिया" and saved nothing.
    r"(?:take|note|write)(?: down)? (?:my|his|her|the) number|number (?:likho|likh lo|note karo))\b"
    r"|कॉल\s*बैक|वापस कॉल|कॉल कर(?:ना|ें|िए|ो)|संपर्क कर|नंबर (?:लिखो|लिख लो|लिखिए|नोट कर)",
    re.I,
)

WANT = re.compile(
    r"\b(?:want|need|like|book|schedule|can i|could i|i'd|please|get|"
    # Romanized Hindi ("appointment chahiye", "milna hai"): a dental parent's booking was missed.
    r"chahiye|chahta|chahti|milega|milegi|mil sakta|mil sakti|karwana|karna hai|kar do|kar dijiye|milna hai|"
    r"dikhana|dikhana hai|dikhwana)\b"
    r"|चाहिए|चाहता|चाहती|मिल सकता|मिल सकती|मिलेगा|मिलेगी|करवा|करा|कर दो|कर दीजिए|कीजिए|बुक|मिलना है|दिखाना|दिखवाना",
    re.I,
)

# "I'm interested" / "interested hoon": a clear yes to the product, even without an
# offer just before it (the LLM kept pitching and never took the details).
INTERESTED = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|great|nice|good|(?:that )?sounds (?:useful|good|great))[,.!]?\s*)*"
    r"(?:i'?m|i am|we'?re|we are|main|hum)\s+(?:very |really |definitely )?interested"
    r"(?:\s+(?:hoon|hu|hain|hai|in (?:it|this|that|the \w+)))?[\s.!]*$"
    r"|^(?:मुझे|हमें)\s+(?:इसमें\s+)?(?:रुचि|इंटरेस्ट)\s+है[\s।!]*$"
    r"|^(?:मैं)\s+(?:इंटरेस्टेड|interested)\s+(?:हूँ|हूं)",
    re.I,
)

# Caller asked to be contacted / gave contact intent in their own words.
CONTACT_WORDS = re.compile(
    r"\b(?:call me|callback|call back|contact me|reach me|my number|interested|sign me up)\b"
    r"|कॉल कर|कॉल बैक|संपर्क कर|बात करवा|मेरा नंबर|interested",
    re.I,
)

HUMAN = re.compile(
    r"\b(?:human|real person|a person|representative|someone from (?:the|your) team|team member|"
    r"manager|executive|customer care)\b|इंसान|असली (?:व्यक्ति|आदमी)|किसी (?:व्यक्ति|आदमी)|"
    r"टीम (?:के किसी|से किसी|मेंबर)|मैनेजर|एग्ज़ीक्यूटिव",
    re.I,
)

# Wanting to talk to someone ("Are you a real person or a bot?" is not a request).
TALK = re.compile(
    r"\b(?:talk|speak|connect|transfer|put me|call|want|need|baat)\b|बात|कनेक्ट|चाहिए|जोड़",
    re.I,
)

OFFER = re.compile(
    r"consultation|demo|meeting|team (?:to )?(?:call|reach|contact)|call you back|"
    # "क्या आप चाहेंगे कि हम एक फ्री डिस्कवरी ऑडिट करें?": the KB's own offer words
    r"\bfree\b|audit|discovery|session|site visit|visit (?:us|our)|trial|appointment|book|"
    r"कंसल्टेशन|डेमो|मीटिंग|टीम.*(?:कॉल|संपर्क|बात)|बुक|फ्री|फ़्री|ऑडिट|डिस्कवरी|सेशन|विज़िट|विजिट|ट्रायल|अपॉइंटमेंट",
    re.I,
)

# The caller volunteers their name ("मेरा नाम हिमांशु है", "my name is Ravi"): they want to
# be contacted, so take the number (the LLM said "nice to meet you" and moved on).
GAVE_NAME = re.compile(
    r"\b(?:my name is|my name's|myself|mera naam|mera nam|merra naam)\b|मेरा नाम|मेरी नाम|"
    r"\b(?:bol raha|bol rahi) (?:hoon|hu|hun)\b|बोल रह[ाी] (?:हूँ|हूं)",
    re.I,
)

# Maya promised or claimed a booking ("मैं कंसल्टेशन बुक कर देती हूँ", "I've booked it")
# before any name/number was taken: the caller's next yes must start the contact flow.
PROMISED = re.compile(
    r"\b(?:i'?ll|i will|let me|i'?m going to|i am going to|i'?ve|i have)\s+(?:go ahead and\s+)?"
    r"(?:book|booked|schedule|scheduled|set up|arrange|arranged)\b|\bbooked (?:a|your|the|it)\b"
    r"|बुक कर (?:देती|देता|रही|रहा|दिया|दी|दूँ|दूं)|बुक हो (?:गया|गई|जाएगा)|शेड्यूल कर",
    re.I,
)


# ---- Ending the call -----------------------------------------------------------
# The caller is clearly ending the call (not just "thanks" mid-conversation).
CLEAR_GOODBYE = re.compile(
    r"\b(?:bye|goodbye|that'?s all|that is all|no thanks?|nothing else|that'?s it|i'?m done|"
    r"wrong number|galti se|nahi bas|bas itna(?: hi)?|aur kuch nahi)\b"
    # A reply that is only "thanks" ("ठीक है धन्यवाद", "ok thank you") is a sign-off.
    r"|^\W*(?:(?:ok(?:ay)?|ठीक है|theek hai|accha|अच्छा)\W*)?(?:thanks?|thank you|धन्यवाद|शुक्रिया|dhanyavaad|shukriya)\W*$"
    r"|बाय|बस इतना|और कुछ नहीं|अलविदा|फ़ोन रखता|फोन रखता|रखती हूँ|रखता हूँ|गलती से|ग़लती से|गलत नंबर",
    re.I,
)

# A bare "no" answering "anything else?" also means the caller is done.
SHORT_NO = re.compile(
    r"^\W*(?:no|nope|nah|no no|nothing|not really|nahi|nahin|नहीं|ना|जी नहीं|नहीं जी|कुछ नहीं|"
    r"नहीं धन्यवाद|no thank you|no that's it|बस|बस इतना ही)\W*$", re.I
)

# Maya's reply already sounds like a goodbye ("आपका दिन शुभ हो!", "have a great day").
SIGNOFF = re.compile(
    r"goodbye|\bbye\b|have a (?:good|nice|great) day|team will (?:contact|call)|"
    r"anything else|शुभ हो|नमस्ते|अलविदा|संपर्क करेगी|कॉल करेगी|और किसी|और कुछ",
    re.I,
)

# Caller is wrapping up: only then may end_call actually hang up.
_GOODBYE_STRONG = re.compile(
    r"\b(?:bye|goodbye|good night|that's all|thats all|that is all|no thanks?|nothing else|"
    r"not now|that's it|thats it|i'm done|im done|hang up|cut the call|"
    r"ok bye|okay bye|rakhta|rakhti|take care|see you|see ya|"
    r"have a (?:good|nice|great) day)\b|बाय|बस इतना|और कुछ नहीं|रखता|रखती",
    re.I,
)

# Polite words that are just as often mid-conversation ("Thanks! And do you build
# apps?", "Alright, tell me more"): a goodbye only when nothing else is asked.
_GOODBYE_SOFT = re.compile(
    r"\b(?:thank you|thanks|thank u|done|chalo|alright|all right|ok then|okay then)\b|धन्यवाद|शुक्रिया|चलो",
    re.I,
)

_STILL_TALKING = re.compile(
    r"\?|\b(?:and|but|also|what|which|how|why|when|where|who|can|could|do|does|tell|more|"
    r"explain|about|need|want|please)\b|और|लेकिन|क्या|कैसे|कब|कहाँ|कौन|बताइए|बताओ|बता|चाहिए|मुझे",
    re.I,
)

class _Goodbye:
    """`GOODBYE.search(text)`: True when the caller's words mean they're done."""

    @staticmethod
    def search(text: str) -> bool:
        if _GOODBYE_STRONG.search(text):
            return True
        if not _GOODBYE_SOFT.search(text):
            return False
        # "Thanks", "Okay thank you", "धन्यवाद जी": short and nothing more asked.
        rest = _GOODBYE_SOFT.sub(" ", text)
        return len(text.split()) <= 5 and not _STILL_TALKING.search(rest)

GOODBYE = _Goodbye()


# ---- Questions and topics ------------------------------------------------------
# Caller is asking something (worth a "let me check" if the lookup is slow).
QUESTION = re.compile(
    r"\?|\b(?:what|which|how|why|when|where|who|can you|could you|tell me|explain|do you|does)\b"
    r"|क्या|कैसे|कौन|कब|कहाँ|क्यों|बताइए|बताओ|बता",
    re.I,
)

# Statements that still need facts ("I want to know about the mill software").
TOPIC = re.compile(
    r"\b(?:about|know|service|services|product|products|software|price|pricing|cost|offer|"
    r"build|create|make|develop|automate|custom|chatbot|app|website|AI|features?|company|contact|"
    r"whatsapp|email|address|located|clients?)\b"
    r"|बारे|सर्विस|प्रोडक्ट|सॉफ्टवेयर|कीमत|कंपनी|company|software",
    re.I,
)

# Pure small talk: no lookup needed.
SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|hii|ok|okay|yes|yeah|yep|no|nope|thanks|thank you|bye|"
    r"hmm+|haan|ha|nahi|theek hai|accha|achha|namaste|हाँ|नहीं|ठीक है|अच्छा|नमस्ते)[\s.!?,]*$",
    re.I,
)

PRICING_QUESTION = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|charge|charges|fee|fees|rate|rates|quote|budget|"
    r"how much|kitna|kitne|keemat|daam|paisa|paise)\b|कीमत|दाम|कितना|कितने|शुल्क",
    re.I,
)

# Questions whose answers aren't in the knowledge base (see llm_node nudges).
UNKNOWN_FACT = re.compile(
    r"\b(?:timing|timings|hours|open|closed|sunday|saturday|holiday|upi|emi|gst|invoice|payment|"
    r"job|jobs|internship|vacancy|salary|kab khula|khula)\b|खुला|बंद|रविवार|शनिवार|छुट्टी|टाइमिंग|"
    r"पेमेंट|ईएमआई|इनवॉइस|नौकरी|इंटर्नशिप|सैलरी",
    re.I,
)

# Office timings / visiting hours: answered by a callback, never by the LLM.
OFFICE_HOURS = re.compile(
    r"\b(?:office (?:hours|timings?)|timings?|opening hours|working hours|open on|closed on|"
    r"(?:open|closed) (?:on )?(?:sunday|saturday|today|tomorrow)|when (?:are you|is the office) open|"
    r"kab khula|kab band|milne aa|kitne baje|khul(?:ta|ti|te) hai|kab khul(?:ta|ti|te)|band (?:hota|rehta|rehti)|"
    # Visiting: "I'd like to visit you", "can I come and meet you", "drop by your office"
    # Visiting the office ("visit you", "visit your office"); "visit the site" is a booking.
    r"visit (?:you|us|your (?:office|shop|clinic|store)|the office)|come (?:to|and|over|by)\b.{0,20}(?:office|meet|see|visit)|"
    r"meet (?:you|the team) (?:in person|at)|drop by|walk in)\b"
    r"|कब खुला|कब बंद|खुला रहता|बंद रहता|टाइमिंग|ऑफिस.*(?:आ सकता|आ सकती|मिलने|खुला|बंद)|मिलने आ|"
    r"कितने बजे|खुलता है|खुलती है|खुलते हैं|कब खुलता|कब खुलती",
    re.I,
)

# The caller also asked where the office is.
WHERE = re.compile(r"\b(?:where|address|location|kahan|kaha|pata)\b|कहाँ|कहां|पता|लोकेशन|एड्रेस", re.I)


# ---- Emergencies -----------------------------------------------------------------
# A caller describing a medical / safety emergency: tell them to call for help now,
# never start a booking (maya.py). Ordinary pain ("दाँत में दर्द") is not one.
EMERGENCY = re.compile(
    r"\b(?:chest pain|heart attack|can'?t breathe|cannot breathe|not breathing|stopped breathing|"
    r"unconscious|fainted|collapsed|heavy bleeding|bleeding (?:a lot|heavily|badly)|stroke|seizure|fits|"
    r"(?:had|met with) an accident|accident (?:ho gaya|hua hai|hua)|on fire|fire in|poison(?:ed|ing)?|"
    r"overdose|suicid\w*|seene (?:mein|me) dard|saans nahi|behosh|khoon beh)\b"
    r"|सीने में (?:बहुत )?दर्द|सीने में जलन और|दिल का दौरा|हार्ट अटैक|साँस नहीं|सांस नहीं|बेहोश|"
    r"खून बह|एक्सीडेंट हो गया|एक्सीडेंट हुआ|दुर्घटना हो|आग लग|ज़हर खा|जहर खा|दौरा पड़",
    re.I,
)


# ---- Other ---------------------------------------------------------------------
# Caller asks Maya to wait ("hold on", "ek minute"): the silence check-in waits longer.
HOLD = re.compile(
    r"\b(?:hold on|wait a (?:minute|second|sec|moment)|please wait|one (?:minute|second|sec)|just a (?:minute|second|sec)|give me a (?:minute|second|sec)|"
    r"let me (?:check|see)|ek (?:minute|min|second|sec)|ruko|rukiye)\b|रुको|रुकिए|एक मिनट|एक सेकंड|देखता हूँ|देखती हूँ",
    re.I,
)
