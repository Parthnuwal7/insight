# LLM judge packet - run `20260812T212944Z-v2-general`

**150 reviews.** The judge is deliberately blind to pipeline output.

## How to use

1. Paste everything between the PROMPT markers below into a strong LLM
   (Claude Opus, GPT-4-class, or similar). One shot, no follow-ups.
2. Save its raw JSON reply to `benchmarks/judgments/20260812T212944Z-v2-general.json`.
3. Score it:

   ```
   .venv-bench/Scripts/python.exe benchmarks/harness/score_judgments.py --run 20260812T212944Z-v2-general
   ```

If the model truncates, split the review list into two halves, run each, then
concatenate the `judgments` arrays into a single JSON object before saving.

---

## ===== PROMPT START =====

You are labelling customer reviews for an aspect-based sentiment analysis benchmark.

For each review, identify every ASPECT the reviewer actually evaluates, and the
sentiment they express toward that specific aspect.

RULES

1. An aspect is a thing being judged (battery, delivery speed, staff attitude,
   onboarding), not a topic the review merely mentions. If the reviewer expresses
   no stance toward it, it is not an aspect.

2. Name each aspect as a short noun phrase, lowercase, using the reviewer's own
   framing where possible. Prefer "delivery" over "the delivery was slow".
   Prefer "battery life" over "battery life issues".

3. One entry per distinct aspect. If the same aspect is praised twice, emit it once.
   If the same aspect is both praised and criticised, emit it once with the
   sentiment that dominates the reviewer's overall stance toward it.

4. Sentiment is strictly one of: Positive, Negative, Neutral.
   Judge the reviewer's INTENT, not surface vocabulary:
   - Negation flips polarity. "does not drain quickly" about battery is Positive.
   - Sarcasm flips polarity. "Brilliant, arrived snapped in half" is Negative.
   - Neutral means genuinely balanced or purely factual, not "mildly positive".

5. Include IMPLICIT aspects. If a reviewer says "already looking for the charger
   twice a day", the aspect is "battery life" with sentiment Negative, even
   though the word battery never appears.

6. For non-English or code-mixed (Hinglish) reviews, label the aspects in English
   but judge sentiment from the original meaning.

7. Evidence must be the shortest verbatim span from the review that supports your
   judgment. Copy it exactly; do not paraphrase.

8. If a review evaluates nothing, return an empty aspects list. Do not invent
   aspects to fill space.

OUTPUT FORMAT

Return ONE JSON object and nothing else. No prose before or after, no markdown
fences. It must validate against this shape:

{
  "judgments": [
    {
      "review_id": <integer, copied exactly from the input>,
      "language": "en" | "hi" | "hi-latn" | "other",
      "aspects": [
        {
          "aspect": "<short lowercase noun phrase>",
          "sentiment": "Positive" | "Negative" | "Neutral",
          "evidence": "<verbatim span from the review>"
        }
      ]
    }
  ]
}

Emit exactly one judgment object per input review, in the same order.


INPUT REVIEWS

```json
{
  "reviews": [
    {
      "review_id": 1,
      "review": "The delivery was incredibly fast, arrived in two days.",
      "title": ""
    },
    {
      "review_id": 2,
      "review": "Packaging was terrible, the box was completely crushed.",
      "title": ""
    },
    {
      "review_id": 3,
      "review": "The price is very reasonable for what you get.",
      "title": ""
    },
    {
      "review_id": 4,
      "review": "Customer service never responded to my emails.",
      "title": ""
    },
    {
      "review_id": 5,
      "review": "The refund was processed without any hassle.",
      "title": ""
    },
    {
      "review_id": 6,
      "review": "Product quality is disappointing for this price point.",
      "title": ""
    },
    {
      "review_id": 7,
      "review": "Great product and the shipping was quick too. Very satisfied.",
      "title": ""
    },
    {
      "review_id": 8,
      "review": "The item arrived damaged and the seller refused a replacement.",
      "title": ""
    },
    {
      "review_id": 9,
      "review": "Excellent packaging, fast delivery, and the product works perfectly.",
      "title": ""
    },
    {
      "review_id": 10,
      "review": "Poor build quality and the price is far too high.",
      "title": ""
    },
    {
      "review_id": 11,
      "review": "The website was easy to navigate and checkout was smooth.",
      "title": ""
    },
    {
      "review_id": 12,
      "review": "Delivery took three weeks and the packaging was falling apart.",
      "title": ""
    },
    {
      "review_id": 13,
      "review": "Good value for money and the return policy is generous.",
      "title": ""
    },
    {
      "review_id": 14,
      "review": "The size chart was inaccurate and customer support was unhelpful.",
      "title": ""
    },
    {
      "review_id": 15,
      "review": "Fast shipping, secure packaging, and a fair price overall.",
      "title": ""
    },
    {
      "review_id": 16,
      "review": "The product is excellent but the delivery was painfully slow.",
      "title": ""
    },
    {
      "review_id": 17,
      "review": "Packaging was great, though the quality did not match the photos.",
      "title": ""
    },
    {
      "review_id": 18,
      "review": "Cheap price, but you get what you pay for. The material feels flimsy.",
      "title": ""
    },
    {
      "review_id": 19,
      "review": "Customer service was very responsive, but the refund took a month.",
      "title": ""
    },
    {
      "review_id": 20,
      "review": "Love the design, hate the price.",
      "title": ""
    },
    {
      "review_id": 21,
      "review": "Shipping was fast but the item was the wrong colour.",
      "title": ""
    },
    {
      "review_id": 22,
      "review": "I ordered this set for my new apartment and I have mixed things to say. The delivery was genuinely impressive, it arrived a full day earlier than estimated. Packaging was sturdy and everything was wrapped individually so nothing was broken. However, the quality of the fabric is nowhere near what the listing photos suggested. It feels thin and I doubt it will survive many washes. Customer service did respond quickly when I complained, which I appreciated, but they were not able to offer a partial refund. The price was reasonable, so I am keeping it, but I would not order again.",
      "title": ""
    },
    {
      "review_id": 23,
      "review": "This was my third order from this seller and sadly the worst. The shipping took nineteen days with no tracking updates for most of that time. When the parcel finally arrived the packaging was torn open at one corner and one of the three items was missing entirely. I contacted customer support and after four days of silence they told me to file a claim myself. The product that did arrive is fine, the build quality is solid and it works as described, but the entire buying experience was exhausting.",
      "title": ""
    },
    {
      "review_id": 24,
      "review": "Genuinely one of the better online purchases I have made this year. The website made it easy to compare options and the size guide was accurate, which is rare. Checkout was quick and I got a confirmation email immediately. Delivery arrived within the promised window and the driver was polite. The product itself exceeded expectations, the material is thick and well finished. The price was higher than competitors but I think the quality justifies it.",
      "title": ""
    },
    {
      "review_id": 25,
      "review": "I want to be fair here because there are good and bad parts. The product arrived on time and the packaging was recyclable which I liked. But the instructions were almost unreadable, poorly translated and missing two steps. Assembly took me three hours instead of the advertised thirty minutes. Once assembled the item is sturdy and looks good in the room. The price is fair for the size. I would buy again but I would watch a video first.",
      "title": ""
    },
    {
      "review_id": 26,
      "review": "Ordering was simple enough but everything after that went wrong. The delivery date moved three times. When it finally came the box was dented and the product inside had a scratch across the front panel. I asked for a replacement and the seller was actually very reasonable about it, they sent a new one within a week at no cost. The replacement is perfect, the finish is flawless and it works well. So the product is good and the seller is honest, but the courier let them down badly.",
      "title": ""
    },
    {
      "review_id": 27,
      "review": "डिलीवरी बहुत तेज़ थी और पैकेजिंग भी अच्छी थी।",
      "title": ""
    },
    {
      "review_id": 28,
      "review": "कीमत बहुत ज़्यादा है और गुणवत्ता खराब है।",
      "title": ""
    },
    {
      "review_id": 29,
      "review": "Delivery bahut fast thi but quality thodi average hai.",
      "title": ""
    },
    {
      "review_id": 30,
      "review": "Packaging ekdum solid tha aur price bhi reasonable hai.",
      "title": ""
    },
    {
      "review_id": 31,
      "review": "The interface is clean and intuitive.",
      "title": ""
    },
    {
      "review_id": 32,
      "review": "Battery drain is severe when the app runs in the background.",
      "title": ""
    },
    {
      "review_id": 33,
      "review": "Sync works flawlessly across all my devices.",
      "title": ""
    },
    {
      "review_id": 34,
      "review": "There are far too many ads in the free version.",
      "title": ""
    },
    {
      "review_id": 35,
      "review": "The dark mode looks fantastic.",
      "title": ""
    },
    {
      "review_id": 36,
      "review": "The app crashes every time I open the settings.",
      "title": ""
    },
    {
      "review_id": 37,
      "review": "Fast performance and a beautiful interface. Highly recommended.",
      "title": ""
    },
    {
      "review_id": 38,
      "review": "The app is slow to load and notifications never arrive on time.",
      "title": ""
    },
    {
      "review_id": 39,
      "review": "Great search function, smooth animations, and offline mode actually works.",
      "title": ""
    },
    {
      "review_id": 40,
      "review": "The subscription is overpriced and the free tier is unusable.",
      "title": ""
    },
    {
      "review_id": 41,
      "review": "Login was simple and the onboarding tutorial was genuinely helpful.",
      "title": ""
    },
    {
      "review_id": 42,
      "review": "Constant crashes and the data sync lost two weeks of my notes.",
      "title": ""
    },
    {
      "review_id": 43,
      "review": "The widget is useful and the customisation options are extensive.",
      "title": ""
    },
    {
      "review_id": 44,
      "review": "Permissions requested are excessive and the privacy policy is vague.",
      "title": ""
    },
    {
      "review_id": 45,
      "review": "Smooth performance, no ads, and the export feature saves me hours.",
      "title": ""
    },
    {
      "review_id": 46,
      "review": "The interface is gorgeous but the app drains my battery in hours.",
      "title": ""
    },
    {
      "review_id": 47,
      "review": "Powerful features, shame about the confusing navigation.",
      "title": ""
    },
    {
      "review_id": 48,
      "review": "The latest update improved performance but removed my favourite widget.",
      "title": ""
    },
    {
      "review_id": 49,
      "review": "Sync is reliable, however the interface feels dated.",
      "title": ""
    },
    {
      "review_id": 50,
      "review": "Customer support replied within an hour but could not fix the crash.",
      "title": ""
    },
    {
      "review_id": 51,
      "review": "Free version is generous, though the ads are intrusive.",
      "title": ""
    },
    {
      "review_id": 52,
      "review": "I have been using this app daily for about eight months so I feel qualified to review it properly. The interface is genuinely one of the best I have used, everything is where you expect it to be and the animations are smooth without being slow. Sync across my phone and tablet has never once failed. That said, battery consumption is a real problem, I lose roughly fifteen percent overnight even with background refresh disabled. The subscription price went up twice this year which felt greedy. Customer support was responsive when I raised the battery issue but their answer was essentially that it is expected behaviour.",
      "title": ""
    },
    {
      "review_id": 53,
      "review": "This started as a great app and has slowly got worse. Two years ago the performance was excellent and there were no ads at all. The last three updates have introduced a banner ad on every screen, and the app now takes nearly ten seconds to open on my device. The search function still works well and I still rely on the offline mode, which is why I have not switched. But the notification system is broken, I get reminders hours late or not at all. I would not recommend it to a new user today.",
      "title": ""
    },
    {
      "review_id": 54,
      "review": "Switched to this from a competitor last month and I am impressed. The onboarding was quick and it imported all my existing data without a single error, which I did not expect. The interface takes a little learning but once you understand the layout it is very efficient. Performance is fast even with thousands of entries. The free tier is limited but fair, and the paid plan is cheaper than what I was using before. Dark mode is well implemented. My only complaint is that the tablet layout wastes a lot of space.",
      "title": ""
    },
    {
      "review_id": 55,
      "review": "Mixed review because the app does one thing brilliantly and everything else poorly. The core editor is fast, stable and a pleasure to use. I have never had it crash while editing. But the cloud sync is unreliable, it has silently failed twice and I only noticed days later. The settings menu is a maze. Customer support took eleven days to reply to a data loss report, which is unacceptable for a paid product. The pricing is reasonable at least.",
      "title": ""
    },
    {
      "review_id": 56,
      "review": "I installed this after seeing it recommended and I regret it. The signup process demanded access to my contacts and location before I could even see the app, which is an immediate red flag. The interface is cluttered with upsell banners. Performance is sluggish, scrolling stutters constantly on a recent phone. The one positive is that the export function works properly and let me get my data back out easily, so uninstalling was painless.",
      "title": ""
    },
    {
      "review_id": 57,
      "review": "ऐप का इंटरफेस बहुत साफ है लेकिन बैटरी जल्दी खत्म होती है।",
      "title": ""
    },
    {
      "review_id": 58,
      "review": "विज्ञापन बहुत ज़्यादा हैं और ऐप बार बार बंद हो जाता है।",
      "title": ""
    },
    {
      "review_id": 59,
      "review": "App ka interface bahut accha hai lekin ads irritating hain.",
      "title": ""
    },
    {
      "review_id": 60,
      "review": "Performance fast hai aur sync bhi perfectly kaam karta hai.",
      "title": ""
    },
    {
      "review_id": 61,
      "review": "The food was absolutely delicious.",
      "title": ""
    },
    {
      "review_id": 62,
      "review": "Service was painfully slow all evening.",
      "title": ""
    },
    {
      "review_id": 63,
      "review": "The ambiance is warm and welcoming.",
      "title": ""
    },
    {
      "review_id": 64,
      "review": "Portions are far too small for the price.",
      "title": ""
    },
    {
      "review_id": 65,
      "review": "The staff were friendly and attentive throughout.",
      "title": ""
    },
    {
      "review_id": 66,
      "review": "The dessert menu is very limited.",
      "title": ""
    },
    {
      "review_id": 67,
      "review": "Wonderful food and excellent service. We will be back.",
      "title": ""
    },
    {
      "review_id": 68,
      "review": "The pasta was undercooked and the waiter was rude.",
      "title": ""
    },
    {
      "review_id": 69,
      "review": "Lovely atmosphere, generous portions, and reasonable prices.",
      "title": ""
    },
    {
      "review_id": 70,
      "review": "The wait time was over an hour and the table was sticky.",
      "title": ""
    },
    {
      "review_id": 71,
      "review": "Excellent wine list and the steak was cooked perfectly.",
      "title": ""
    },
    {
      "review_id": 72,
      "review": "Noisy dining room and the music was far too loud to talk.",
      "title": ""
    },
    {
      "review_id": 73,
      "review": "The bread was fresh and the olive oil was excellent quality.",
      "title": ""
    },
    {
      "review_id": 74,
      "review": "Overpriced menu and the parking situation is a nightmare.",
      "title": ""
    },
    {
      "review_id": 75,
      "review": "Fresh ingredients, beautiful presentation, and friendly staff.",
      "title": ""
    },
    {
      "review_id": 76,
      "review": "The food was outstanding but the service was incredibly slow.",
      "title": ""
    },
    {
      "review_id": 77,
      "review": "Beautiful decor, shame the food was bland.",
      "title": ""
    },
    {
      "review_id": 78,
      "review": "Staff were lovely but the kitchen got our order wrong twice.",
      "title": ""
    },
    {
      "review_id": 79,
      "review": "Great value for money, although the seating is uncomfortable.",
      "title": ""
    },
    {
      "review_id": 80,
      "review": "The curry was superb, however the naan arrived cold.",
      "title": ""
    },
    {
      "review_id": 81,
      "review": "Quick service but the coffee tasted burnt.",
      "title": ""
    },
    {
      "review_id": 82,
      "review": "We booked here for an anniversary dinner and it mostly lived up to expectations. The ambiance is genuinely special, low lighting and well spaced tables so you can actually hold a conversation. Our server was attentive without hovering and knew the menu well. The starters were the highlight, the scallops in particular were cooked perfectly. The main course was less impressive, my lamb was noticeably overcooked and dry. The wine list is extensive but the markup is steep. Dessert made up for it and the staff brought out a candle without us asking.",
      "title": ""
    },
    {
      "review_id": 83,
      "review": "Sadly this place has gone downhill since it changed hands. We used to come monthly. The menu has been cut in half and the prices have gone up noticeably. The food quality is not what it was, my risotto was gluey and clearly reheated. Service was disorganised, we waited twenty minutes to order and our drinks arrived after the food. The dining room itself is still lovely and the location is convenient, which is the only reason I would consider going back.",
      "title": ""
    },
    {
      "review_id": 84,
      "review": "Genuinely one of the best meals I have had this year. Everything from start to finish was considered. The bread arrived warm with excellent butter. The tasting menu was well paced and each course was distinct. Portions were judged well, I left satisfied but not uncomfortable. The staff were knowledgeable and clearly enjoyed working there. Prices are high but honestly justified given the quality of the ingredients. The only small criticism is that the restaurant is quite hard to find.",
      "title": ""
    },
    {
      "review_id": 85,
      "review": "Went for a casual weekday lunch and it was fine, nothing more. The food was competent, my sandwich was fresh and the soup was properly seasoned. Service was efficient and we were in and out in forty minutes which suited us. The interior is a bit tired, the chairs are worn and the lighting is harsh. Prices are on the higher side for what is essentially a cafe. It does the job if you need a quick lunch nearby.",
      "title": ""
    },
    {
      "review_id": 86,
      "review": "Booked a table for eight for a birthday and the restaurant handled it badly. Despite booking three weeks ahead they had us at two separate tables. The manager was apologetic and did eventually rearrange things, which I appreciated. Once seated the food was very good, the shared platters especially were generous and full of flavour. Service after that point was attentive. But the noise level made conversation across the table impossible and the bill took half an hour to arrive.",
      "title": ""
    },
    {
      "review_id": 87,
      "review": "खाना बहुत स्वादिष्ट था और सेवा भी अच्छी थी।",
      "title": ""
    },
    {
      "review_id": 88,
      "review": "कीमत बहुत ज़्यादा थी और माहौल शोरगुल वाला था।",
      "title": ""
    },
    {
      "review_id": 89,
      "review": "Khana bahut tasty tha lekin service slow thi.",
      "title": ""
    },
    {
      "review_id": 90,
      "review": "Ambiance ekdum mast tha aur staff bhi friendly the.",
      "title": ""
    },
    {
      "review_id": 91,
      "review": "The room was spotlessly clean.",
      "title": ""
    },
    {
      "review_id": 92,
      "review": "The wifi was unusable for the entire stay.",
      "title": ""
    },
    {
      "review_id": 93,
      "review": "Breakfast was excellent with plenty of choice.",
      "title": ""
    },
    {
      "review_id": 94,
      "review": "The bathroom was outdated and poorly maintained.",
      "title": ""
    },
    {
      "review_id": 95,
      "review": "The location is perfect for exploring the city.",
      "title": ""
    },
    {
      "review_id": 96,
      "review": "The beds were extremely uncomfortable.",
      "title": ""
    },
    {
      "review_id": 97,
      "review": "Lovely room and the staff were incredibly helpful.",
      "title": ""
    },
    {
      "review_id": 98,
      "review": "The air conditioning was broken and reception was unresponsive.",
      "title": ""
    },
    {
      "review_id": 99,
      "review": "Comfortable beds, quiet rooms, and an excellent breakfast spread.",
      "title": ""
    },
    {
      "review_id": 100,
      "review": "The pool was closed and the gym equipment was broken.",
      "title": ""
    },
    {
      "review_id": 101,
      "review": "Check in was fast and the room had a stunning view.",
      "title": ""
    },
    {
      "review_id": 102,
      "review": "Thin walls, noisy corridors, and the heating never worked.",
      "title": ""
    },
    {
      "review_id": 103,
      "review": "The spa was wonderful and the restaurant served great food.",
      "title": ""
    },
    {
      "review_id": 104,
      "review": "Parking is expensive and the lift was out of service all week.",
      "title": ""
    },
    {
      "review_id": 105,
      "review": "Spacious room, modern bathroom, and very friendly reception staff.",
      "title": ""
    },
    {
      "review_id": 106,
      "review": "The location is excellent but the rooms are very dated.",
      "title": ""
    },
    {
      "review_id": 107,
      "review": "Staff were wonderful, though the breakfast was disappointing.",
      "title": ""
    },
    {
      "review_id": 108,
      "review": "Beautiful lobby, but our room smelled of damp.",
      "title": ""
    },
    {
      "review_id": 109,
      "review": "Great value for the price, although the wifi kept dropping.",
      "title": ""
    },
    {
      "review_id": 110,
      "review": "The bed was extremely comfortable but the shower had no pressure.",
      "title": ""
    },
    {
      "review_id": 111,
      "review": "The garden was quiet and peaceful, however the check out process was chaotic.",
      "title": ""
    },
    {
      "review_id": 112,
      "review": "Stayed here for four nights on a work trip and it was a solid choice. The location is the standout feature, five minutes from the station and walking distance to everything I needed. Check in was efficient even though I arrived late. The room was compact but very well designed, with good storage and a genuinely comfortable bed. The bathroom was modern and the shower had excellent pressure. Breakfast was the weak point, a limited buffet that ran out of most things by half eight. Wifi was fast and stable throughout, which matters when you are working.",
      "title": ""
    },
    {
      "review_id": 113,
      "review": "I would not stay here again. The photographs online are clearly several years old. Our room was tired, the carpet was stained and there was visible mould around the window frame. I raised it with reception and they moved us, which was handled politely, but the second room had the same damp smell. The heating was uncontrollable, either off or sweltering. Breakfast was actually decent, fresh fruit and good coffee. The location is convenient but not enough to make up for the state of the rooms.",
      "title": ""
    },
    {
      "review_id": 114,
      "review": "A genuinely lovely place for a weekend away. The building has real character and the staff clearly care about it. Our room overlooked the garden and was quiet all night. The bed and pillows were excellent. Breakfast is cooked to order rather than a buffet which made a big difference to the quality. The spa was small but immaculate and never crowded. Parking is limited and we had to use a public car park nearby, which was the only inconvenience. Prices are fair for what you get.",
      "title": ""
    },
    {
      "review_id": 115,
      "review": "Booked this for a family holiday and it was a mixed bag. The pool area is fantastic, clean and well supervised, and the children loved it. Our family room was spacious enough for four with proper beds rather than fold outs. However the restaurant was consistently poor, the food was lukewarm every evening and the choice for children was limited to chips. Housekeeping was inconsistent, some days thorough and some days clearly skipped. The reception staff were always friendly and helped us book excursions.",
      "title": ""
    },
    {
      "review_id": 116,
      "review": "Arrived to find our booking had not been recorded despite having a confirmation email. The receptionist was apologetic and found us a room, but it was a downgrade from what we paid for and no refund was offered at the time. The room itself was clean and the bed comfortable. Noise from the street was significant, we could hear traffic all night through single glazed windows. Breakfast was fine, standard continental. The hotel did eventually refund the difference after I emailed, so credit to them for resolving it.",
      "title": ""
    },
    {
      "review_id": 117,
      "review": "कमरा बहुत साफ था और स्टाफ मददगार था।",
      "title": ""
    },
    {
      "review_id": 118,
      "review": "बाथरूम गंदा था और नाश्ता बहुत खराब था।",
      "title": ""
    },
    {
      "review_id": 119,
      "review": "Room ekdum clean tha lekin wifi bilkul kaam nahi kar raha tha.",
      "title": ""
    },
    {
      "review_id": 120,
      "review": "Location bahut acchi hai aur breakfast bhi tasty tha.",
      "title": ""
    },
    {
      "review_id": 121,
      "review": "The battery life is outstanding, easily two full days.",
      "title": ""
    },
    {
      "review_id": 122,
      "review": "The screen scratches far too easily.",
      "title": ""
    },
    {
      "review_id": 123,
      "review": "Sound quality is rich and well balanced.",
      "title": ""
    },
    {
      "review_id": 124,
      "review": "The charging cable stopped working after two weeks.",
      "title": ""
    },
    {
      "review_id": 125,
      "review": "The build quality feels premium and solid.",
      "title": ""
    },
    {
      "review_id": 126,
      "review": "The camera struggles badly in low light.",
      "title": ""
    },
    {
      "review_id": 127,
      "review": "Excellent display and the battery lasts all day.",
      "title": ""
    },
    {
      "review_id": 128,
      "review": "The speakers are tinny and the microphone picks up too much noise.",
      "title": ""
    },
    {
      "review_id": 129,
      "review": "Fast processor, sharp screen, and the fingerprint sensor is reliable.",
      "title": ""
    },
    {
      "review_id": 130,
      "review": "The device overheats during use and the fan is very loud.",
      "title": ""
    },
    {
      "review_id": 131,
      "review": "Great keyboard feel and the trackpad is very precise.",
      "title": ""
    },
    {
      "review_id": 132,
      "review": "The software is bloated and storage fills up within months.",
      "title": ""
    },
    {
      "review_id": 133,
      "review": "Lightweight design, long battery, and it charges very quickly.",
      "title": ""
    },
    {
      "review_id": 134,
      "review": "The port selection is limited and the included adapter feels cheap.",
      "title": ""
    },
    {
      "review_id": 135,
      "review": "Crisp display, excellent speakers, and a very responsive touchscreen.",
      "title": ""
    },
    {
      "review_id": 136,
      "review": "The camera is superb but the battery drains within four hours.",
      "title": ""
    },
    {
      "review_id": 137,
      "review": "Beautiful screen, shame about the flimsy hinge.",
      "title": ""
    },
    {
      "review_id": 138,
      "review": "Performance is excellent, although the fan noise is distracting.",
      "title": ""
    },
    {
      "review_id": 139,
      "review": "The price is very competitive but the build feels cheap.",
      "title": ""
    },
    {
      "review_id": 140,
      "review": "Sound quality is great, however the bluetooth connection keeps dropping.",
      "title": ""
    },
    {
      "review_id": 141,
      "review": "Setup was straightforward but the manual was useless.",
      "title": ""
    },
    {
      "review_id": 142,
      "review": "I have had this laptop for roughly six months of daily development work so this is a considered review. The build quality is genuinely excellent, the chassis has no flex and the hinge still feels tight. The keyboard is the best I have used on a portable machine, good travel and no rattle. Performance handles everything I throw at it including containers and a couple of virtual machines. Battery life is the weak point, I get around four hours under real load rather than the advertised ten. The fan becomes quite loud under sustained compilation. The display is sharp and colour accurate which matters for my work.",
      "title": ""
    },
    {
      "review_id": 143,
      "review": "Returned this after ten days and I want to explain why. The screen is genuinely beautiful, bright and with excellent contrast, and the speakers are far better than I expected. But the software experience ruined it. The device shipped with a large amount of preinstalled software I could not remove, and the interface lagged when switching between apps despite the powerful processor. Storage was already forty percent used out of the box. The camera was mediocre in anything other than bright daylight. For the price I expected considerably better.",
      "title": ""
    },
    {
      "review_id": 144,
      "review": "Upgraded from a five year old model and the difference is significant. Setup took under ten minutes and it transferred everything across automatically. The display is a huge improvement, much brighter and the refresh rate makes scrolling feel smooth. Battery comfortably lasts a full day of heavy use with charge to spare. The cameras are excellent, particularly the ultrawide. Build quality feels solid and the weight is well balanced. My only reservation is the price, which is high, and the charger is no longer included in the box.",
      "title": ""
    },
    {
      "review_id": 145,
      "review": "These headphones are a mixed proposition. The sound quality is genuinely very good, detailed with controlled bass that does not overwhelm. Noise cancellation is effective on a plane and on public transport. However the comfort is poor for me, the clamping force is strong and after about ninety minutes my ears ache. The companion app is buggy and disconnects regularly. Battery life is strong at around thirty hours. The carrying case is well made and compact.",
      "title": ""
    },
    {
      "review_id": 146,
      "review": "Bought this monitor for photo editing and it mostly delivers. Colour accuracy out of the box was better than expected and needed only minor calibration. The panel is uniform with no visible backlight bleed. The stand is sturdy and adjusts easily. However the on screen menu system is genuinely awful, navigating it with the small joystick is frustrating and options are buried. The built in speakers are poor but I did not buy it for those. Cable management on the stand is a nice touch.",
      "title": ""
    },
    {
      "review_id": 147,
      "review": "बैटरी बहुत अच्छी है और कैमरा भी शानदार है।",
      "title": ""
    },
    {
      "review_id": 148,
      "review": "स्क्रीन खराब है और कीमत बहुत ज़्यादा है।",
      "title": ""
    },
    {
      "review_id": 149,
      "review": "Camera quality bahut acchi hai lekin battery jaldi khatam hoti hai.",
      "title": ""
    },
    {
      "review_id": 150,
      "review": "Sound bahut clear hai aur build quality bhi solid hai.",
      "title": ""
    }
  ]
}
```

## ===== PROMPT END =====
