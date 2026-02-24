#!/usr/bin/env python3
"""Diagnostic: verify MedGemma 1.5 4B on Vertex is actually processing images."""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dotenv import load_dotenv
load_dotenv()
from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

adapter = VertexMedGemmaAdapter(
    project_id='gen-lang-client-0517724223',
    region='us-central1',
    endpoint_id='923518299076034560',
    model_name='medgemma-4b-vertex',
    gpu_hourly_rate=1.15,
)

IMG = 'ingest_design/MedPix-2-0/images/MPX1452_synpic58265.png'

async def main():
    # Test 1: Text-only
    print('=== Test 1: TEXT-ONLY ===')
    r1 = await adapter.generate(
        'Clinical history: 21 y.o. asymptomatic man with incidental finding on chest radiograph. What is the most likely diagnosis?',
        max_tokens=150
    )
    print(f'Response: {r1.text}')
    print(f'Latency: {r1.latency_ms:.0f}ms, Tokens: {r1.output_tokens}\n')

    # Test 2: Image + simple Q
    print('=== Test 2: IMAGE + simple Q ===')
    r2 = await adapter.generate_with_image(
        prompt='Describe what you see in this medical image in 2-3 sentences.',
        image_path=IMG, max_tokens=200
    )
    print(f'Response: {r2.text}')
    print(f'Latency: {r2.latency_ms:.0f}ms, Tokens: {r2.output_tokens}\n')

    # Test 3: Image + benchmark prompt
    print('=== Test 3: IMAGE + benchmark prompt ===')
    r3 = await adapter.generate_with_image(
        prompt='Clinical history: 21 y.o. asymptomatic man with an incidental finding on a screening chest radiograph.\n\nBased on the image and clinical history, provide:\n\nDIAGNOSIS: [single primary diagnosis]\n\nFINDINGS: [key imaging findings]',
        image_path=IMG, max_tokens=500
    )
    print(f'Response: {r3.text}')
    print(f'Latency: {r3.latency_ms:.0f}ms, Tokens: {r3.output_tokens}\n')

    # Test 4: Image + system message
    print('=== Test 4: IMAGE + system message ===')
    r4 = await adapter.generate_with_image(
        prompt='Clinical history: 21 y.o. asymptomatic man with an incidental finding on a screening chest radiograph.\n\nBased on the image and clinical history, provide:\n\nDIAGNOSIS: [single primary diagnosis]\n\nFINDINGS: [key imaging findings]',
        image_path=IMG, max_tokens=500,
        system_message='You are a board-certified radiologist. Analyze the medical image carefully.'
    )
    print(f'Response: {r4.text}')
    print(f'Latency: {r4.latency_ms:.0f}ms, Tokens: {r4.output_tokens}\n')

asyncio.run(main())
