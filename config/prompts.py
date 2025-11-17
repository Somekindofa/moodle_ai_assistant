from langchain.prompts import PromptTemplate

hyde_prompt = PromptTemplate(input_variables=[], template="""Your goal is to create a **HyDE (Hypothetical Document Embedding)** prompt that generates a believable, detailed, and contextually rich document about the process of glassblowing. This document will serve as a reference to compare against your existing transcripts in your vector store.

Here’s how you can structure your HyDE prompt to ensure the LLM generates a high-quality, realistic document:

---

### **HyDE Prompt for Glassblowing Process**
**Task:**
Write a detailed, first-person narrative describing the step-by-step process of glassblowing, focusing on the techniques, tools, and safety precautions involved. The narrative should be rich in sensory details, technical accuracy, and expert advice, as if written by an experienced glassblower. Include common mistakes, expert tips, and the reasoning behind each action.

**Guidelines:**
1. **First-Person Perspective:**
   Write as if you are an experienced glassblower explaining your process. Use "I" and "my" to create a personal, immersive narrative.

2. **Technical Accuracy:**
   - Describe the tools (e.g., *canne*, *bloc mouillé*, *pince de préhension*) and their purpose.
   - Explain the importance of temperature control, body posture, and hand movements.
   - Mention the physical properties of glass (e.g., malléabilité, fragilité, reaction to temperature changes).

3. **Sensory Details:**
   - Include descriptions of sounds (e.g., the hiss of steam, the clink of tools), smells (e.g., heated glass, wood), and visuals (e.g., the glow of molten glass, the shape of the paraison).
   - Describe the tactile experience (e.g., the resistance of the glass, the heat radiating from the furnace).

4. **Safety Precautions:**
   - Emphasize the importance of protective gear (e.g., heat-resistant gloves, goggles).
   - Discuss body positioning to avoid burns and accidents.
   - Mention ventilation and the risks of inhaling fumes.

5. **Common Mistakes and Expert Tips:**
   - Highlight errors beginners often make (e.g., gripping the cane too tightly, overheating the glass).
   - Share advice from experts (e.g., how to judge the ideal temperature, how to maintain consistent pressure).

6. **Flow and Continuity:**
   - Structure the narrative as a continuous process, from selecting the cane to shaping the glass and returning to the furnace.
   - Use transitions to connect each step logically.

7. **Authenticity:**
   - Avoid overly technical jargon unless explained.
   - Use natural language and occasional colloquialisms to mimic a real artisan’s voice.

---

### **Example Output Structure**
```markdown
Je commence ma journée dans l’atelier en préparant mentalement chaque étape du soufflage du verre. La première chose que je fais est de sélectionner une canne parmi celles préchauffées dans les petits fourneaux. La pince de préhension doit être tenue avec une ferme douceur : assez pour contrôler le verre, mais jamais assez pour le comprimer et risquer de le fissurer. Mes pouces s’opposent à mes doigts, créant une prise stable, tandis que je garde mon corps légèrement en retrait pour éviter les brûlures. Une erreur classique ici est de saisir la canne trop brusquement, ce qui peut provoquer des chocs thermiques et des fissures. Les artisans expérimentés insistent sur l’importance d’un mouvement fluide et contrôlé, presque comme une danse avec le matériau.

En me dirigeant vers l’établi, je pose la canne sur les supports, en vérifiant qu’elle est parfaitement stable et parallèle au sol. Je saisis alors le bloc mouillé, un outil en bois trempé dans l’eau, dont la vapeur forme un coussin protecteur entre le verre et le bloc. Ce coussin permet de centrer et façonner la paraison sans laisser de marques. Je fais tourner la canne lentement, en ajustant la pression pour éviter les déformations. La fumée qui s’élève est cette vapeur essentielle, un signe que tout se passe comme prévu. Trop de pression ou une rotation trop rapide, et la forme en souffrira.

Quand je retourne au four pour réchauffer le verre, je reste conscient de ma posture : pieds écartés, dos droit, et toujours une distance de sécurité. Les gants résistants à la chaleur protègent mes mains, mais c’est l’observation de la couleur et de la consistance du verre qui me guide. Un verre trop fluide est aussi difficile à maîtriser qu’un verre trop froid. Les experts disent qu’il faut "écouter" le verre, sentir sa résistance et sa malléabilité pour savoir quand il est prêt.

Chaque étape est une leçon de patience et de précision. La ventilation de l’atelier est cruciale pour éviter d’inhaler les particules fines, et la pratique régulière affine la coordination entre les mains et le corps. C’est un métier où chaque détail compte, et où l’expérience se transmet autant par les gestes que par les mots.
```

---

### **Why This Works for HyDE**
- **Richness:** The document is dense with technical and sensory details, making it a strong candidate for embedding and comparison.
- **Authenticity:** The first-person perspective and expert tips mimic real-world transcripts.
- **Contextual Depth:** It covers tools, techniques, safety, and common pitfalls, ensuring a comprehensive reference.

Would you like me to refine this further for a specific focus (e.g., beginner vs. advanced techniques, historical vs. modern methods)? Or would you like to test this prompt with your LLM?""")

