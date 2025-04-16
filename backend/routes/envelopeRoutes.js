const express = require('express');
const router = express.Router();
const { createEnvelope, getAllEnvelopes, getEnvelopeById, updateEnvelope, deleteEnvelope, transferAmount } = require('../controllers/envelopeController')

router.post('/envelopes', createEnvelope)

router.get('/envelopes', getAllEnvelopes)

router.get('/envelopes/:id', getEnvelopeById)

// PUT route to update a specific envelope by its ID
router.put('/envelopes/:id', updateEnvelope);  // New route to update an envelope

// DELETE route to remove a specific envelope by ID
router.delete('/envelopes/:id', deleteEnvelope);

router.post('/envelopes/transfer/:fromId/:toId', transferAmount);

const fetch = require('node-fetch');

router.post('/envelopes/forecast', async (req, res) => {
    const { category } = req.body;

    try {
        const response = await fetch('http://localhost:5000/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category })
        });
        const data = await response.json();
        res.status(200).json(data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch prediction' });
    }
});


module.exports = router;
