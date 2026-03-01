curl -X POST http://192.168.1.18:8000/generate/image \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Generate an image, without any text, to serve as a background for the presentation slides of my company, which specializes in the following: Renovations and Systems * Complete Renovations * Turnkey Bathrooms and Kitchens * Plumbing, Electrical, Gas, and Air Conditioning (Certified) Finishings and Decorations * Interior/Exterior Painting * Decorative Paintings * Painting of Windows and Furniture * Wallpapering * Facade Renovation Drywall and Insulation * Walls and Ceilings * Soundproofing * Plastering and Stuccoing * Thermal Insulation and Waterproofing"}' \
    --output image.png

