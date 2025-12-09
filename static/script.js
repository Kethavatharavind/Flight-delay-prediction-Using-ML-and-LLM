/**
 * Modern Flight Delay Predictor - Interactive JavaScript
 * ✅ FIXED: Better data validation and error handling
 */

// State Management
const state = {
    routes: [],
    selectedOrigin: null,
    selectedDest: null,
    selectedDate: null,
    flights: [],
    minDate: null,
    maxDate: null,
    loading: false
};

// DOM Elements
const elements = {
    originSelect: document.getElementById('origin-iata'),
    destSelect: document.getElementById('dest-iata'),
    dateInput: document.getElementById('flight-date'),
    dateHint: document.getElementById('date-hint'),
    findFlightsBtn: document.getElementById('find-flights-btn'),
    btnText: document.querySelector('.btn-text'),
    btnLoader: document.querySelector('.btn-loader'),
    errorMessage: document.getElementById('error-message'),
    flightListCard: document.getElementById('flight-list-card'),
    flightList: document.getElementById('flight-list'),
    noFlightsMessage: document.getElementById('no-flights-message'),
    selectedRoute: document.getElementById('selected-route'),
    selectedDateSpan: document.getElementById('selected-date'),
    toast: document.getElementById('toast')
};

// Utility Functions
const showToast = (message, icon = '✅', duration = 3000) => {
    const toast = elements.toast;
    const toastIcon = toast.querySelector('.toast-icon');
    const toastMessage = toast.querySelector('.toast-message');
    
    toastIcon.textContent = icon;
    toastMessage.textContent = message;
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, duration);
};

const showError = (message) => {
    elements.errorMessage.textContent = message;
    elements.errorMessage.classList.remove('hidden');
    setTimeout(() => {
        elements.errorMessage.classList.add('hidden');
    }, 5000);
};

const hideError = () => {
    elements.errorMessage.classList.add('hidden');
};

const setLoading = (isLoading) => {
    state.loading = isLoading;
    
    if (isLoading) {
        elements.findFlightsBtn.disabled = true;
        elements.btnText.classList.add('hidden');
        elements.btnLoader.classList.remove('hidden');
    } else {
        elements.findFlightsBtn.disabled = false;
        elements.btnText.classList.remove('hidden');
        elements.btnLoader.classList.add('hidden');
    }
};

const formatTime = (timeStr) => {
    if (!timeStr || timeStr === 'N/A') return 'N/A';
    return timeStr;
};

const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
};

// Initialize Application
const init = async () => {
    console.log('🚀 Initializing Flight Predictor...');
    
    // Load date range
    await loadDateRange();
    
    // Load routes
    await loadRoutes();
    
    // Setup event listeners
    setupEventListeners();
    
    console.log('✅ Initialization complete');
    showToast('System ready!', '✅', 2000);
};

// Load Date Range
const loadDateRange = async () => {
    try {
        const response = await fetch('/get_min_date');
        const data = await response.json();
        
        state.minDate = data.min_date;
        state.maxDate = data.max_date;
        
        elements.dateInput.min = state.minDate;
        elements.dateInput.max = state.maxDate;
        elements.dateInput.value = state.minDate;
        
        elements.dateHint.textContent = `${state.minDate} to ${state.maxDate}`;
        
        console.log(`📅 Date range: ${state.minDate} to ${state.maxDate}`);
    } catch (error) {
        console.error('❌ Failed to load date range:', error);
        showError('Failed to load date configuration');
    }
};

// Load Routes
const loadRoutes = async () => {
    try {
        const response = await fetch('/get_available_routes');
        if (!response.ok) throw new Error('Failed to load routes');
        
        const data = await response.json();
        state.routes = data;
        
        populateRouteSelects();
        console.log(`✅ Loaded ${state.routes.length} routes`);
    } catch (error) {
        console.error('❌ Failed to load routes:', error);
        showError('Failed to load available routes. Please refresh the page.');
    }
};

// Populate Route Selects
const populateRouteSelects = () => {
    // Get unique origins
    const origins = [...new Set(state.routes.map(r => r.ORIGIN))].sort();
    
    elements.originSelect.innerHTML = '<option value="">Select origin...</option>';
    origins.forEach(origin => {
        const option = document.createElement('option');
        option.value = origin;
        option.textContent = origin;
        elements.originSelect.appendChild(option);
    });
    
    elements.originSelect.disabled = false;
};

const updateDestinations = (origin) => {
    if (!origin) {
        elements.destSelect.innerHTML = '<option value="">Select origin first</option>';
        elements.destSelect.disabled = true;
        return;
    }
    
    const destinations = state.routes
        .filter(r => r.ORIGIN === origin)
        .map(r => r.DEST)
        .sort();
    
    elements.destSelect.innerHTML = '<option value="">Select destination...</option>';
    destinations.forEach(dest => {
        const option = document.createElement('option');
        option.value = dest;
        option.textContent = dest;
        elements.destSelect.appendChild(option);
    });
    
    elements.destSelect.disabled = false;
};

// Setup Event Listeners
const setupEventListeners = () => {
    // Origin selection
    elements.originSelect.addEventListener('change', (e) => {
        state.selectedOrigin = e.target.value;
        updateDestinations(state.selectedOrigin);
        updateFindButton();
    });
    
    // Destination selection
    elements.destSelect.addEventListener('change', (e) => {
        state.selectedDest = e.target.value;
        updateFindButton();
    });
    
    // Date selection
    elements.dateInput.addEventListener('change', (e) => {
        state.selectedDate = e.target.value;
        updateFindButton();
    });
    
    // Find flights button
    elements.findFlightsBtn.addEventListener('click', handleFindFlights);
};

const updateFindButton = () => {
    const isValid = state.selectedOrigin && state.selectedDest && state.selectedDate;
    elements.findFlightsBtn.disabled = !isValid || state.loading;
};

// Find Flights Handler
const handleFindFlights = async () => {
    hideError();
    setLoading(true);
    
    try {
        const response = await fetch('/find_flights', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                origin_iata: state.selectedOrigin,
                dest_iata: state.selectedDest,
                date: state.selectedDate
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to find flights');
        }
        
        state.flights = data.flights || [];
        renderFlightList();
        
        showToast(`Found ${state.flights.length} flight(s)`, '✈️');
        
    } catch (error) {
        console.error('❌ Error finding flights:', error);
        showError(error.message);
    } finally {
        setLoading(false);
    }
};

// Render Flight List
const renderFlightList = () => {
    // Update header
    elements.selectedRoute.textContent = `${state.selectedOrigin} → ${state.selectedDest}`;
    elements.selectedDateSpan.textContent = formatDate(state.selectedDate);
    
    // Show card
    elements.flightListCard.classList.remove('hidden');
    
    // Clear previous flights
    elements.flightList.innerHTML = '';
    
    if (state.flights.length === 0) {
        elements.noFlightsMessage.classList.remove('hidden');
        elements.flightList.classList.add('hidden');
        return;
    }
    
    elements.noFlightsMessage.classList.add('hidden');
    elements.flightList.classList.remove('hidden');
    
    // Render each flight
    state.flights.forEach((flight, index) => {
        const flightElement = createFlightElement(flight, index);
        elements.flightList.appendChild(flightElement);
    });
    
    // Scroll to flight list
    setTimeout(() => {
        elements.flightListCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
};

// Create Flight Element
const createFlightElement = (flight, index) => {
    const div = document.createElement('div');
    div.className = 'flight-item';
    div.style.animationDelay = `${index * 0.1}s`;
    
    div.innerHTML = `
        <div class="flight-header">
            <div>
                <div class="flight-number">${flight.airline_iata} ${flight.flnr}</div>
                <div class="airline-name">${flight.airline}</div>
            </div>
        </div>
        <div class="flight-times">
            <div class="time-info">
                <div class="time-label">Departure</div>
                <div class="time-value">${formatTime(flight.departure_time_display)}</div>
            </div>
            <div class="flight-arrow">✈️</div>
            <div class="time-info">
                <div class="time-label">Arrival</div>
                <div class="time-value">${formatTime(flight.arrival_time_display)}</div>
            </div>
        </div>
    `;
    
    // Click handler - navigate to dashboard
    div.addEventListener('click', () => {
        navigateToDashboard(flight);
    });
    
    return div;
};

// Navigate to Dashboard - ✅ FIXED
const navigateToDashboard = (flight) => {
    // Prepare prediction data
    const predictionData = {
        origin_iata: state.selectedOrigin,
        dest_iata: state.selectedDest,
        date: state.selectedDate,
        flight_number: flight.flnr,
        departure_time: flight.departure_time_scheduled_local,
        arrival_time: flight.arrival_time_scheduled_local,
        airline: flight.airline,
        airline_iata: flight.airline_iata
    };
    
    // ✅ ADDED: Validate data before storing
    console.log('📦 Storing flight data:', predictionData);
    
    // Validate required fields
    if (!predictionData.flight_number) {
        showError('Invalid flight number. Please try again.');
        return;
    }
    
    if (!predictionData.departure_time || !predictionData.arrival_time) {
        showError('Invalid flight times. Please try again.');
        return;
    }
    
    if (!predictionData.origin_iata || !predictionData.dest_iata) {
        showError('Invalid route information. Please try again.');
        return;
    }
    
    // Store in sessionStorage
    try {
        sessionStorage.setItem('flightPredictionData', JSON.stringify(predictionData));
        
        // ✅ ADDED: Verify storage
        const stored = sessionStorage.getItem('flightPredictionData');
        if (!stored) {
            throw new Error('Failed to store data');
        }
        
        const parsed = JSON.parse(stored);
        if (parsed.flight_number !== predictionData.flight_number) {
            throw new Error('Data verification failed');
        }
        
        console.log('✅ Data stored and verified successfully');
        
    } catch (error) {
        console.error('❌ Storage error:', error);
        showError('Failed to save flight data. Please try again.');
        return;
    }
    
    // Show loading indicator
    showToast('Loading prediction...', '🔄', 1500);
    
    // Navigate after brief delay
    setTimeout(() => {
        window.location.href = '/dashboard';
    }, 500);
};

// Error Handler
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    showError('An unexpected error occurred');
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

console.log('✅ Flight Predictor script loaded');
